#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train the memristor models (SGPR1: state, SGPR2: current), export them to Verilog-A,
run HSPICE with the decks in HSPICE_train/ and HSPICE_test/, and collect the training time, the HSPICE
runtime, and the errors.

  python3 experiments/run_memristor_sweep.py --Z 50 200 600
  python3 experiments/run_memristor_sweep.py --model gpr
  python3 experiments/run_memristor_sweep.py --Z 50 200 600 --kernels RBF Matern32 Matern52 RQ
  python3 experiments/run_memristor_sweep.py --Z 50 200 600 --stop none
(run in the root of this repository, or give the root with --root)

The training settings are those of sgpr_train_state.py and sgpr_train_current.py
(inputs, scaling, inducing-point layout, RBF kernel with a length scale for each input,
noise 1e-4 / 1e-3, jitter 1e-6 doubled on failure, Adam lr=0.1, at most 1000 / 200 iterations).
  --stop neg   : stop when loss < 0 (same as sgpr_train_*.py)       [default]
  --stop full  : run all the iterations
  --stop none  : no optimization; only W is calculated in closed form
  --model gpr  : GPR without sparsity. The same
                 loop as the SGPR is applied to pyro.contrib.gp.models.GPRegression, and all the
                 training points are stored in Verilog-A in place of the inducing points.
Each run is written to runs/<tag>/ (verilog/, HSPICE_train/, HSPICE_test/) and one line per run
is appended to the csv given by --out.
The errors are calculated as in utils/plot_MLmodel.py (RMSE and RMSLE of the current between
the measured data and the HSPICE result, row by row).
--dry-run skips HSPICE and uses a Python emulation of rram_gp.va at a time step of 1 ns.
"""
import argparse, csv, os, re, shutil, socket, subprocess, sys, time
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
SCALE_I = 10 ** 4                                   # sgpr_train_current.py ; I_out / pow(10,4) in rram_gp.va
SETTING = {"state": dict(noise=1e-4, epoc=1000), "current": dict(noise=1e-3, epoc=200)}
VA_NAME = {"state": "S", "current": "CURRENT"}       # macro suffix in the Verilog-A files
VA_FN = {"state": "s", "current": "current"}         # function suffix in gp_utils_*.va

KERNEL_VA = {
    "RBF": "    /* RBF kernel */\\\n    kxx[i] = variance * exp(-0.5 * kxx[i]);\\\n",
    "Matern32": "    /* Matern 3/2 kernel */\\\n    kxx[i] = sqrt(3.0 * (kxx[i] + 1.0e-12));\\\n"
                "    kxx[i] = variance * (1.0 + kxx[i]) * exp(-kxx[i]);\\\n",
    "Matern52": "    /* Matern 5/2 kernel */\\\n    kxx[i] = sqrt(5.0 * (kxx[i] + 1.0e-12));\\\n"
                "    kxx[i] = variance * (1.0 + kxx[i] + kxx[i] * kxx[i] / 3.0) * exp(-kxx[i]);\\\n",
    "RQ": "    /* rational quadratic kernel */\\\n"
          "    kxx[i] = variance * pow(1.0 + 0.5 * kxx[i] / `ALPHA_%s, -`ALPHA_%s);\\\n",
}


def kernel_np(name, r2, variance, alpha):
    if name == "RBF":
        return variance * np.exp(-0.5 * r2)
    if name == "Matern32":
        r = np.sqrt(3.0 * (r2 + 1.0e-12)); return variance * (1.0 + r) * np.exp(-r)
    if name == "Matern52":
        r = np.sqrt(5.0 * (r2 + 1.0e-12)); return variance * (1.0 + r + r * r / 3.0) * np.exp(-r)
    if name == "RQ":
        return variance * np.power(1.0 + 0.5 * r2 / alpha, -alpha)
    raise ValueError(name)


def training_data(which, train_file):
    import torch
    d = pd.read_csv(train_file)
    if which == "state":                            # sgpr_train_state.py
        change = d[d["S_in"] != d["S"]]
        d = pd.concat([d, change], ignore_index=True)        # = DataFrame.append in the repository
        X = torch.stack((torch.Tensor(d["S_in"].values), torch.Tensor(d["Voltage"].values)), dim=1)
        y = torch.Tensor(d["S"].values * 1.0)
    else:                                           # sgpr_train_current.py
        X = torch.stack((torch.Tensor(d["S"].values), torch.Tensor(d["Voltage"].values)), dim=1)
        y = torch.Tensor(d["Current"].values * SCALE_I)
    return X, y


def train_one(which, model_type, kname, z, stop, train_file, threads):
    import torch, pyro
    import pyro.contrib.gp as gp
    sys.path.insert(0, ROOT)
    from utils.sgpr_cls import SparseGPRegression as SGPR
    torch.set_num_threads(threads)
    pyro.set_rng_seed(0); pyro.clear_param_store()
    X, y = training_data(which, train_file)
    cfg = SETTING[which]
    kernel = {"RBF": gp.kernels.RBF, "Matern32": gp.kernels.Matern32, "Matern52": gp.kernels.Matern52,
              "RQ": gp.kernels.RationalQuadratic}[kname](input_dim=2, lengthscale=torch.tensor([1.0, 1.0]))
    if model_type == "sgpr":
        xu = torch.Tensor([[0.0, v] for v in np.linspace(-1.5, 1.5, z // 2)]
                          + [[1.0, v] for v in np.linspace(-1.5, 1.0, z // 2)])
        m = SGPR(X, y, kernel, Xu=xu, jitter=1e-6, noise=X.new_tensor(cfg["noise"]))
    else:
        m = gp.models.GPRegression(X, y, kernel, noise=X.new_tensor(cfg["noise"]), jitter=1e-6)

    def weights():
        """W of the predictive mean, mean(x*) = k(x*, points) W"""
        if model_type == "sgpr":
            m.cache_preds(); return m.tmp2_cached.detach()
        with torch.no_grad():
            K = m.kernel(X).contiguous()
            K.view(-1)[::len(X) + 1] += m.jitter + m.noise
            L = torch.linalg.cholesky(K)
            return torch.cholesky_solve(y.unsqueeze(-1), L).squeeze(-1)

    def retry(e):
        m.jitter = m.jitter * 2                     # as in the repository
        return float(m.jitter) > 1.0

    optimizer = torch.optim.Adam(m.parameters(), lr=0.1)
    loss_fn = pyro.infer.TraceMeanField_ELBO().differentiable_loss
    iters = fails = 0; last = float("nan")
    start = time.time()
    if stop != "none":
        for _ in range(cfg["epoc"]):
            try:
                optimizer.zero_grad()
                loss = loss_fn(m.model, m.guide)
                loss.backward()
                optimizer.step()
                iters += 1; last = loss.item()
                if model_type == "sgpr":
                    m.cache_preds()                 # as in the repository
                if stop == "neg" and last < 0:
                    break
            except (ValueError, RuntimeError) as e:
                fails += 1
                if retry(e): break
    while True:                                     # W (included in the training time)
        try:
            W = weights(); break
        except (ValueError, RuntimeError) as e:
            fails += 1
            if retry(e): raise
    train_time = time.time() - start
    pts = (m.Xu if model_type == "sgpr" else X).detach().numpy().astype(float)
    alpha = float(m.kernel.scale_mixture.detach()) if kname == "RQ" else 1.0
    model = dict(kernel=kname, xu=pts, w=W.numpy().astype(float).reshape(-1),
                 lengthscale=m.kernel.lengthscale.detach().numpy().astype(float),
                 variance=float(m.kernel.variance.detach()), alpha=alpha)
    info = dict(N=len(X), iters=iters, fails=fails, loss=last, time=train_time,
                noise=float(m.noise.detach()), jitter=float(torch.as_tensor(m.jitter)))
    return model, info


def export(models, run_dir):
    vdir = os.path.join(run_dir, "verilog"); os.makedirs(vdir, exist_ok=True)
    shutil.copy(os.path.join(ROOT, "verilog", "rram_gp.va"), vdir)
    for which, m in models.items():
        U, f = VA_NAME[which], VA_FN[which]
        with open(os.path.join(vdir, "variables_%s.va" % which), "w") as o:
            o.write("`define N_%s %d\n`define X_DIM_%s 2\n\n`define LENGTHSCALE_%s {\\\n%s\\\n}\n\n"
                    "`define VARIANCE_%s %r\n`define ALPHA_%s %r\n"
                    % (U, len(m["xu"]), U, U, ",\\\n".join(repr(float(v)) for v in m["lengthscale"]),
                       U, m["variance"], U, m["alpha"]))
        with open(os.path.join(vdir, "xu_%s.va" % which), "w") as o:
            o.write("`define XU_%s {\\\n%s\\\n}\n" % (U, ",\\\n".join(",".join(repr(float(v)) for v in x) for x in m["xu"])))
        with open(os.path.join(vdir, "w_vec_%s.va" % which), "w") as o:
            o.write("// woodbury vector\n`define W_VEC_%s {\\\n%s\\\n};\n" % (U, ",\\\n".join(repr(float(v)) for v in m["w"])))
        src = open(os.path.join(ROOT, "verilog", "gp_utils_%s.va" % which)).read()
        old = "    /* RBF kernel */\\\n    kxx[i] = variance * exp(-0.5 * kxx[i]);\\\n"
        if src.count(old) != 1:
            sys.exit("the kernel line was not found in verilog/gp_utils_%s.va" % which)
        new = KERNEL_VA[m["kernel"]]
        if m["kernel"] == "RQ": new = new % (U, U)
        open(os.path.join(vdir, "gp_utils_%s.va" % which), "w").write(src.replace(old, new))
    for d in ("HSPICE_train", "HSPICE_test"):
        os.makedirs(os.path.join(run_dir, d, "output"), exist_ok=True)
        shutil.copy(os.path.join(ROOT, d, "1R.sp"), os.path.join(run_dir, d))


def run_hspice(deck_dir, cmd):
    with open(os.path.join(deck_dir, "output", "hspice_stdout.txt"), "w") as log:
        return subprocess.call(cmd.split() + ["-i", "1R.sp", "-o", "output/output"],
                               cwd=deck_dir, stdout=log, stderr=subprocess.STDOUT)


def read_lis(fn):
    """time [ns], V(BL,Nb), I(v_t1), V(X1.S_out) ; and the runtime printed by HSPICE"""
    rows = []; started = False; runtime = float("nan")
    with open(fn, errors="ignore") as f:
        for line in f:
            if "transient analysis" in line:
                started = True; continue
            m = re.search(r"job total runtime\s+([\d.eE+-]+)", line)
            if m: runtime = float(m.group(1))
            if started and "job concluded" not in line:
                p = line.split()
                if len(p) == 4:
                    try: rows.append([float(v) for v in p])
                    except ValueError: pass
    if not rows:
        raise RuntimeError("no transient data in " + fn)
    a = np.array(rows)
    return pd.DataFrame({"Time": a[:, 0] * 1e9, "S": a[:, 3], "Voltage": a[:, 1], "Current": a[:, 2]}), runtime


def diagnose(deck_dir, ret):
    print("  !! HSPICE did not produce transient data (return code %s) : %s" % (ret, deck_dir))
    for name in ("hspice_stdout.txt", "output.lis", "output.valog"):
        fn = os.path.join(deck_dir, "output", name)
        if not os.path.exists(fn):
            print("     %s : not created" % name); continue
        lines = open(fn, errors="ignore").read().split("\n")
        hits = [l for l in lines if re.search(r"error|abort|fail|cannot|unable|not found|lic:|licens|\*pvaE\*", l, re.I)]
        print("     ---- %s (%d lines)" % (name, len(lines)))
        for l in hits[:25]: print("     | " + l[:200])
        for l in lines[-10:]: print("     . " + l[:200])


# ---------------------------------------------------------------- Python emulation (for --dry-run)
def deck_voltage(deck):
    """V(BL,Nb) of 1R.sp at every 1 ns : sum of the V_SET pulses minus the sum of the V_RESET pulses"""
    txt = open(deck).read()
    par = {k: float(v) for k, v in re.findall(r"\.param\s+(\w+)\s*=\s*([\d.]+)V", txt)}
    tstop = float(re.search(r"\.tran\s+\S+\s+([\d.]+)ns", txt).group(1))
    t = np.arange(0.0, tstop + 0.5, 1.0); v = np.zeros_like(t)
    for name, v1, td, tr, tf, pw, per in re.findall(
            r"^(V_\w+)\s+\S+\s+\S+\s+PULSE\s*\(\s*0V\s+(\w+)\s+([\d.]+)ns\s+([\d.]+)ns\s+([\d.]+)ns\s+([\d.]+)ns\s+([\d.]+)ns", txt, re.M):
        a = par[v1]; td, tr, tf, pw, per = map(float, (td, tr, tf, pw, per))
        x = np.mod(t - td, per); x[t < td] = -1.0
        p = np.where(x < 0, 0.0, np.where(x < tr, a * x / tr, np.where(x < tr + pw, a, np.where(x < tr + pw + tf, a * (1 - (x - tr - pw) / tf), 0.0))))
        v += p if name.upper().startswith("V_SET") else -p
    return t, v


def gp_mean(m, x):
    d = (np.asarray(x)[None, :] - m["xu"]) / m["lengthscale"][None, :]
    return float(kernel_np(m["kernel"], (d ** 2).sum(-1), m["variance"], m["alpha"]) @ m["w"])


def emulate(models, deck):
    t, v = deck_voltage(deck)
    S = 1.0; out = []
    for ti, vi in zip(t, v):
        S = 0.0 if gp_mean(models["state"], [S, vi]) < 0.5 else 1.0
        out.append([ti, S, vi, gp_mean(models["current"], [S, vi]) / SCALE_I])
    return pd.DataFrame(out, columns=["Time", "S", "Voltage", "Current"])


def errors(meas_file, sim):
    """utils/plot_MLmodel.py : cmp_rmse and cmp_rmsle (pandas aligns the two series by the row number)"""
    y = pd.read_csv(meas_file)["Current"]; p = sim["Current"]
    rmse = float(np.sqrt(np.mean((y - p) ** 2)))
    rmsle = float(np.sqrt(np.mean((np.log1p(np.abs(y)) - np.log1p(np.abs(p))) ** 2)))
    return rmse * 1e3, rmsle * 1e3, len(y), len(p)          # [mA], x1000 as in log_hspice.txt


def main():
    global ROOT
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None, help="root of this repository (default: current directory or the parent of the script directory)")
    ap.add_argument("--model", default="sgpr", choices=["sgpr", "gpr"])
    ap.add_argument("--kernels", nargs="+", default=["RBF"], choices=list(KERNEL_VA))
    ap.add_argument("--Z", nargs="+", type=int, default=[50, 200, 600])
    ap.add_argument("--stop", nargs="+", default=["neg"], choices=["neg", "full", "none"])
    ap.add_argument("--train", default="./data/train_TriD_real_new2.csv")
    ap.add_argument("--test", default="./data/test_RanD_real_new2.csv")
    ap.add_argument("--threads", type=int, default=10)
    ap.add_argument("--hspice", default="hspice")
    ap.add_argument("--repeat", type=int, default=1, help="number of HSPICE runs (the runtime is averaged)")
    ap.add_argument("--train-repeat", type=int, default=3,
                    help="number of trainings of each model. The training is deterministic (the seed is reset), so the "
                         "models are identical; train_time is the median and train_time_first is the first one, which "
                         "includes the start-up of the libraries when it is the first training of the process")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--suffix", default="")
    ap.add_argument("--out", default="./results/memristor_sweep.csv")
    a = ap.parse_args()
    for c in ([a.root] if a.root else [os.getcwd(), ROOT, os.path.dirname(ROOT)]):
        c = os.path.abspath(os.path.expanduser(c))
        if os.path.exists(os.path.join(c, "utils", "sgpr_cls.py")) and os.path.exists(os.path.join(c, "verilog", "rram_gp.va")):
            ROOT = c; break
    else:
        sys.exit("The root of this repository was not found. Give its directory with --root.")
    os.chdir(ROOT)
    out = os.path.abspath(os.path.expanduser(a.out)); os.makedirs(os.path.dirname(out), exist_ok=True)
    import torch, pyro
    fields = ["run", "model", "kernel", "Z", "stop", "sim", "host", "threads", "torch", "pyro",
              "N_state", "N_current", "points_state", "points_current", "iters_state", "iters_current",
              "fails_state", "fails_current", "loss_state", "loss_current", "train_state", "train_current", "train_time", "train_time_first", "train_repeat",
              "runtime_train", "runtime_test", "rmse_train_mA", "rmsle_train_x1000", "rmse_test_mA", "rmsle_test_x1000",
              "rows_meas_test", "rows_sim_test", "state_mismatch_test"]
    new = not os.path.exists(out)
    with open(out, "a", newline="") as fo:
        wr = csv.DictWriter(fo, fieldnames=fields)
        if new: wr.writeheader()
        if a.model == "gpr": a.Z = [0]
        for kname in a.kernels:
            for z in a.Z:
                for stop in a.stop:
                    tag = (("GPR_%s_%s" % (kname, stop)) if a.model == "gpr" else "%s_Z%d_%s" % (kname, z, stop)) + a.suffix
                    run_dir = os.path.join(ROOT, "runs", tag)
                    print("==== %s : training" % tag, flush=True)
                    models, info = {}, {}
                    try:
                        for which in ("state", "current"):
                            ts = []
                            for _ in range(max(1, a.train_repeat)):
                                models[which], info[which] = train_one(which, a.model, kname, z, stop, a.train, a.threads)
                                ts.append(info[which]["time"])
                            info[which]["first"] = ts[0]; info[which]["time"] = float(np.median(ts))
                        if not all(np.isfinite(models[w]["w"]).all() for w in models):
                            raise RuntimeError("W is not finite")
                    except (ValueError, RuntimeError) as e:
                        print("  !! training failed (%s). This run is skipped." % str(e).split("\n")[0][:100], flush=True)
                        wr.writerow(dict(run=tag, model=a.model, kernel=kname, Z=z, stop=stop, sim="training failed",
                                         host=socket.gethostname(), threads=a.threads)); fo.flush()
                        continue
                        print("  %-7s N=%d points=%d iters=%d time=%.3fs loss=%.1f" % (which, info[which]["N"], len(models[which]["xu"]),
                              info[which]["iters"], info[which]["time"], info[which]["loss"]), flush=True)
                    export(models, run_dir)
                    sims, rts = {}, {}
                    for d, meas in (("HSPICE_train", a.train), ("HSPICE_test", a.test)):
                        deck_dir = os.path.join(run_dir, d)
                        if a.dry_run:
                            sims[d] = emulate(models, os.path.join(deck_dir, "1R.sp")); rts[d] = float("nan")
                        else:
                            r = []
                            for _ in range(a.repeat):
                                ret = run_hspice(deck_dir, a.hspice)
                                try:
                                    sims[d], rt = read_lis(os.path.join(deck_dir, "output", "output.lis"))
                                except (RuntimeError, FileNotFoundError):
                                    diagnose(deck_dir, ret); sys.exit(1)
                                r.append(rt)
                            rts[d] = float(np.mean(r))
                        sims[d].to_csv(os.path.join(deck_dir, "output", "sim_data.csv"), index=False)
                    e_tr = errors(a.train, sims["HSPICE_train"]); e_te = errors(a.test, sims["HSPICE_test"])
                    s_meas = pd.read_csv(a.test)["S"]; s_sim = sims["HSPICE_test"]["S"]
                    n = min(len(s_meas), len(s_sim))
                    mism = int((np.round(s_meas.values[:n]) != np.round(s_sim.values[:n])).sum())
                    row = dict(run=tag, model=a.model, kernel=kname, Z=z, stop=stop, sim="emulation" if a.dry_run else "hspice",
                               host=socket.gethostname(), threads=a.threads, torch=torch.__version__, pyro=pyro.__version__,
                               N_state=info["state"]["N"], N_current=info["current"]["N"],
                               points_state=len(models["state"]["xu"]), points_current=len(models["current"]["xu"]),
                               iters_state=info["state"]["iters"], iters_current=info["current"]["iters"],
                               fails_state=info["state"]["fails"], fails_current=info["current"]["fails"],
                               loss_state=info["state"]["loss"], loss_current=info["current"]["loss"],
                               train_state=info["state"]["time"], train_current=info["current"]["time"],
                               train_time=info["state"]["time"] + info["current"]["time"],
                               train_time_first=info["state"]["first"] + info["current"]["first"], train_repeat=max(1, a.train_repeat),
                               runtime_train=rts["HSPICE_train"], runtime_test=rts["HSPICE_test"],
                               rmse_train_mA=e_tr[0], rmsle_train_x1000=e_tr[1], rmse_test_mA=e_te[0], rmsle_test_x1000=e_te[1],
                               rows_meas_test=e_te[2], rows_sim_test=e_te[3], state_mismatch_test=mism)
                    wr.writerow(row); fo.flush()
                    print("  train %.3fs | runtime test %s s | train RMSE %.4f mA RMSLE %.4f | test RMSE %.4f mA RMSLE %.4f | state mismatch %d"
                          % (row["train_time"], ("%.2f" % row["runtime_test"]) if not a.dry_run else "-", e_tr[0], e_tr[1], e_te[0], e_te[1], mism), flush=True)
    print("results: " + out)


if __name__ == "__main__":
    main()
