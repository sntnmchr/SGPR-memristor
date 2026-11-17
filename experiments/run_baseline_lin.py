#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Conventional method of the public repository RRAMunif (github.com/albertlin11/RRAMunif):
train the model of
    Codeupload/Real RRAM/RRAM_discrete_classification_real_new_modifyAlb5_SPICEsupple.py
(a feed-forward network: 2-20-20-2 for the state and 3-20-20-1 for the logarithm of the current),
export it to Verilog-A in the form of Codeupload/Real RRAM_Verilog-A_rev/hspice_ML/rram.va, and run
HSPICE with the same decks (input waveforms) as the SGPR models of this repository.

  python3 experiments/run_baseline_lin.py --root /path/to/RRAMunif-main

The training part (data, "SPICE supplement", sample weights, logarithm of the current, scaling,
network, optimizer, batch size 32, 10000 epochs, seed 1) is copied from the script of the repository.
The training time is the time of model.fit(). The progress output of Keras is turned off (verbose=0).
The errors are calculated as in utils/plot_MLmodel.py.
--dry-run skips HSPICE and uses a Python emulation of the Verilog-A model at a time step of 1 ns.
"""
import argparse, csv, os, random, re, shutil, socket, subprocess, sys, time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from run_memristor_sweep import deck_voltage, diagnose          # noqa: E402


# ------------------------------------------------------------------ functions of the repository script
def swCalc_IW(i, w):
    swii = np.ones((i.shape[0],))
    swww = np.ones((w.shape[0],))
    for ii in range(i.shape[0]):
        if np.absolute(i[ii]) > 8e-6:
            swii[ii] = 2
        if np.absolute(i[ii]) > 1.75e-4:
            swii[ii] = 36
        if np.absolute(i[ii]) > 7e-4:
            swii[ii] = 138
    for ii in range(w.shape[0] - 1):
        kk = 38
        if w[ii] != w[ii + 1]:
            swww[ii] = 138
            swww[ii - 1] = kk
            swww[ii - 2] = kk
            swww[ii + 1] = kk
            swww[ii + 2] = kk
    return swww.reshape((-1,)), swii.reshape((-1,))


def logtrans(a):
    for ii in range(a.shape[0]):
        if a[ii] >= 0:
            a[ii] = np.log10(a[ii])
        else:
            a[ii] = np.log10(-a[ii])
    return a


def train(root, epochs, threads, seed_value=1):
    os.environ["PYTHONHASHSEED"] = str(seed_value)
    random.seed(seed_value); np.random.seed(seed_value)
    import tensorflow as tf
    from sklearn.preprocessing import MinMaxScaler
    tf.config.threading.set_intra_op_parallelism_threads(threads)
    tf.random.set_seed(seed_value)

    df = pd.read_csv(os.path.join(root, "Codeupload", "Real RRAM", "input data", "IV_RRAM_TriD_35_1_real_new2.csv"))
    w_train = np.array(df.iloc[:, 0]).reshape(-1, 1).astype(float)
    w_train_0 = np.concatenate((np.array(df.iloc[:1, 0]), np.array(df.iloc[:-1, 0])), axis=0).reshape(-1, 1).astype(float)
    v_train = np.array(df.iloc[:, 1]).reshape(-1, 1)
    y_train = np.array(df.iloc[:, 2]).reshape(-1, 1)
    n_meas = len(v_train)
    # SPICE supplement
    for ind in range(w_train_0.shape[0]):
        if v_train[ind, 0] > 0.8 or v_train[ind, 0] < -1.2:
            v_train = np.concatenate((v_train, v_train[ind, 0].reshape((-1, 1))), axis=0)
            w_train_0 = np.concatenate((w_train_0, 1 - w_train_0[ind, 0].reshape((-1, 1))), axis=0)
            y_train = np.concatenate((y_train, y_train[ind, 0].reshape((-1, 1))), axis=0)
            w_train = np.concatenate((w_train, w_train[ind, 0].reshape((-1, 1))), axis=0)
    sww, swi = swCalc_IW(y_train, w_train_0)
    y_train = logtrans(y_train)
    scaler1 = MinMaxScaler(); scaler2 = MinMaxScaler()
    v_train = scaler1.fit_transform(v_train)
    y_train = scaler2.fit_transform(y_train)

    v_next = tf.keras.Input(shape=(1,), name="Voltage")
    w_init = tf.keras.Input(shape=(1,), name="State")
    w_v_init = tf.keras.layers.Concatenate(axis=1)([v_next, w_init])
    w_v_init = tf.keras.layers.Dense(units=20, activation="relu")(w_v_init)
    w_next = tf.keras.layers.Dense(units=20, activation="relu")(w_v_init)
    w_next = tf.keras.layers.Dense(units=2, activation="sigmoid", name="output_w")(w_next)
    w_v_next = tf.keras.layers.Concatenate(axis=1)([v_next, w_next])
    w_v_next = tf.keras.layers.Dense(units=20, activation="relu")(w_v_next)
    y_next = tf.keras.layers.Dense(units=20, activation="relu")(w_v_next)
    y_next = tf.keras.layers.Dense(units=1, name="output_y")(y_next)
    model = tf.keras.Model(inputs=[v_next, w_init], outputs=[w_next, y_next])
    model.compile(optimizer="adam", loss=["sparse_categorical_crossentropy", "mse"], loss_weights=[0.5, 0.5],
                  metrics=["accuracy", "mse"])
    callback = tf.keras.callbacks.EarlyStopping(monitor="loss", mode="min", patience=300000, verbose=1, restore_best_weights=True)

    class Progress(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if (epoch + 1) % 500 == 0:
                print("    epoch %d  loss %.5f  (%.0f s)" % (epoch + 1, logs["loss"], time.time() - start), flush=True)

    start = time.time()
    hist = model.fit(x=[v_train, w_train_0], y=[w_train, y_train], sample_weight=[sww, swi], batch_size=32,
                     epochs=epochs, shuffle=True, verbose=0, callbacks=[callback, Progress()])
    train_time = time.time() - start
    weights = [np.asarray(w, dtype=float) for w in model.get_weights()]
    shapes = [w.shape for w in weights]
    expect = [(2, 20), (20,), (20, 20), (20,), (20, 2), (2,), (3, 20), (20,), (20, 20), (20,), (20, 1), (1,)]
    if shapes != expect:
        sys.exit("unexpected order of the weights: %s" % shapes)
    net = dict(w=weights, v_min=float(scaler1.data_min_[0]), v_range=float(scaler1.data_range_[0]),
               y_min=float(scaler2.data_min_[0]), y_range=float(scaler2.data_range_[0]))
    info = dict(train_time=train_time, epochs=len(hist.history["loss"]), loss=float(hist.history["loss"][-1]),
                n_meas=n_meas, n_train=len(v_train), tf=tf.__version__)
    return net, info


# ------------------------------------------------------------------ Verilog-A (same form as hspice_ML/rram.va)
def vec(name, v):
    return " real %s[0:%d]={%s};\n" % (name, len(v) - 1, ", ".join(repr(float(x)) for x in v))


def export_va(net, fn):
    W = net["w"]
    o = ['`include "constants.vams"\n`include "disciplines.vams"\n\nmodule linear(g, s);\n'
         "parameter real\t\ttime_step\t= 1e-9 from(1e-15: 20);\n\n inout         g,s;   // external nodes\n\n real W,Wa,Wb;\n\n real A;\n"]
    o += [vec("A%d" % k, W[0][k, :]) for k in range(2)] + [vec("B0", W[1]), " real Ja[0:19];\n\n real B;\n"]
    o += [vec("C%d" % k, W[2][k, :]) for k in range(20)] + [vec("D0", W[3]), " real Jb[0:19];\n\n"]
    o += [vec("E%d" % k, W[4][:, k]) for k in range(2)] + [vec("F0", W[5]), "\n real C;\n"]
    o += [vec("G%d" % k, W[6][k, :]) for k in range(3)] + [vec("H0", W[7]), " real Jc[0:19];\n\n real D;\n"]
    o += [vec("I%d" % k, W[8][k, :]) for k in range(20)] + [vec("J0", W[9]), " real Jd[0:19];\n\n"]
    o += [vec("K0", W[10][:, 0]), " real L0=%r;\n\n real Vgs,Idaa,Idab,Idb;\n real h;\n\n" % float(W[11][0])]
    sumB = "+".join("C%d[h]*Ja[%d]" % (k, k) for k in range(20))
    sumD = "+".join("I%d[h]*Jc[%d]" % (k, k) for k in range(20))
    o.append(""" analog begin

 // bound time step
 $bound_step(time_step);

 @(initial_step("tran")) begin
   W=1;
 end

 Vgs = V(g,s);

 // normalization
 Vgs = (Vgs-(%(vmin)r))/%(vrange)r;

 // MLP model one
 for(h=0; h<=19; h=h+1) begin
  A = A0[h]*Vgs+A1[h]*W+B0[h];
  if (A <= 0)
   Ja[h] = 0;
  else if (A > 0)
   Ja[h] = A;
 end

 for(h=0; h<=19; h=h+1) begin
  B = %(sumB)s+D0[h];
  if (B <= 0)
   Jb[h] = 0;
  else if (B > 0)
   Jb[h] = B;
 end

 Idaa = 0;
 for(h=0; h<=19; h=h+1) begin
  Idaa = Idaa+Jb[h]*E0[h];
 end
 Wa =Idaa+F0[0];
 Wa = (-1)*Wa;
 Wa = 1 / (1+exp(Wa));

 Idab = 0;
 for(h=0; h<=19; h=h+1) begin
  Idab = Idab+Jb[h]*E1[h];
 end
 Wb =Idab+F0[1];
 Wb = (-1)*Wb;
 Wb = 1 / (1+exp(Wb));

 // MLP model two
 for(h=0; h<=19; h=h+1) begin
  C = G0[h]*Vgs+G1[h]*Wa+G2[h]*Wb+H0[h];
  if (C <= 0)
   Jc[h] = 0;
  else if (C > 0)
   Jc[h] = C;
 end

 for(h=0; h<=19; h=h+1) begin
  D = %(sumD)s+J0[h];
  if (D <= 0)
   Jd[h] = 0;
  else if (D > 0)
   Jd[h] = D;
 end

 Idb = 0;
 for(h=0; h<=19; h=h+1) begin
  Idb = Idb+Jd[h]*K0[h];
 end

 Idb = (Idb+L0)*(%(yrange)r)+(%(ymin)r);

 Vgs = (Vgs*%(vrange)r)+(%(vmin)r);

 if (Vgs >= 0)
  Idb = pow(10, Idb);
 else if (Vgs < 0)
  Idb = (-1)*pow(10, Idb);

 if (Wa <= Wb)
  W = 1;
 else if (Wa > Wb)
  W = 0;

 I(g,s) <+ Idb;
 end
endmodule
""" % dict(vmin=net["v_min"], vrange=net["v_range"], ymin=net["y_min"], yrange=net["y_range"], sumB=sumB, sumD=sumD))
    open(fn, "w").write("".join(o))


def load_stored(root):
    """weights of Codeupload/Real RRAM_Verilog-A_rev/hspice_ML/rram.va (the model trained by the authors of the repository)"""
    fn = os.path.join(root, "Codeupload", "Real RRAM_Verilog-A_rev", "hspice_ML", "rram.va")
    t = open(fn).read()
    arr = {m.group(1): np.array([float(x) for x in m.group(2).split(",")])
           for m in re.finditer(r"real (\w+)\[0:\d+\]\s*=\s*\{([^}]*)\}", t)}
    L0 = float(re.search(r"real L0=([-\d.e]+)", t).group(1))
    vn = re.search(r"Vgs = \(Vgs\+([\d.]+)\)/([\d.]+);", t); yn = re.search(r"Idb = \(Idb\+L0\)\*\(([\d.]+)\)-([\d.]+);", t)
    W = [np.stack([arr["A0"], arr["A1"]]), arr["B0"], np.stack([arr["C%d" % k] for k in range(20)]), arr["D0"],
         np.stack([arr["E0"], arr["E1"]], axis=1), arr["F0"], np.stack([arr["G0"], arr["G1"], arr["G2"]]), arr["H0"],
         np.stack([arr["I%d" % k] for k in range(20)]), arr["J0"], arr["K0"].reshape(20, 1), np.array([L0])]
    net = dict(w=W, v_min=-float(vn.group(1)), v_range=float(vn.group(2)), y_min=-float(yn.group(2)), y_range=float(yn.group(1)))
    return net, fn


def forward(net, v, w):
    """one evaluation of the Verilog-A model : returns the current and the next state"""
    W = net["w"]; x = (v - net["v_min"]) / net["v_range"]
    a = np.maximum(0.0, W[0][0] * x + W[0][1] * w + W[1])
    b = np.maximum(0.0, a @ W[2] + W[3])
    s = 1.0 / (1.0 + np.exp(-(b @ W[4] + W[5])))
    c = np.maximum(0.0, W[6][0] * x + W[6][1] * s[0] + W[6][2] * s[1] + W[7])
    d = np.maximum(0.0, c @ W[8] + W[9])
    y = (float(d @ W[10][:, 0]) + float(W[11][0])) * net["y_range"] + net["y_min"]
    cur = 10.0 ** y if v >= 0 else -(10.0 ** y)
    return cur, (1.0 if s[0] <= s[1] else 0.0)


def emulate(net, deck):
    t, v = deck_voltage(deck); w = 1.0; out = []
    for ti, vi in zip(t, v):
        cur, w = forward(net, vi, w)
        out.append([ti, w, vi, cur])
    return pd.DataFrame(out, columns=["Time", "S", "Voltage", "Current"])


def make_deck(src, dst):
    txt = open(src).read()
    txt, a = re.subn(r"\.hdl\s+'\.\./verilog/rram_gp\.va'", ".hdl '../verilog/rram.va'", txt)
    txt, b = re.subn(r"X1\s+BL\s+Nb\s+rram\s*\n\+tstep=1ns", "X1  BL  Nb  linear\n+time_step=1n", txt)
    txt, c = re.subn(r"\.print V\(BL,Nb\) I\(v_t1\) V\(X1\.S_out\)", ".print V(BL,Nb) I(v_t1)", txt)
    if (a, b, c) != (1, 1, 1):
        sys.exit("the deck %s does not have the expected form %s" % (src, (a, b, c)))
    open(dst, "w").write(txt)


def read_lis(fn):
    rows = []; started = False; runtime = float("nan")
    with open(fn, errors="ignore") as f:
        for line in f:
            if "transient analysis" in line:
                started = True; continue
            m = re.search(r"job total runtime\s+([\d.eE+-]+)", line)
            if m: runtime = float(m.group(1))
            if started and "job concluded" not in line:
                p = line.split()
                if len(p) == 3:
                    try: rows.append([float(v) for v in p])
                    except ValueError: pass
    if not rows:
        raise RuntimeError("no transient data in " + fn)
    a = np.array(rows)
    return pd.DataFrame({"Time": a[:, 0] * 1e9, "Voltage": a[:, 1], "Current": a[:, 2]}), runtime


def errors(meas_file, sim):
    y = pd.read_csv(meas_file)["Current"]; p = sim["Current"]
    rmse = float(np.sqrt(np.mean((y - p) ** 2)))
    rmsle = float(np.sqrt(np.mean((np.log1p(np.abs(y)) - np.log1p(np.abs(p))) ** 2)))
    return rmse * 1e3, rmsle * 1e3, len(y), len(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="RRAMunif-main")
    ap.add_argument("--mr-root", default=os.path.dirname(HERE), help="root of this repository (decks and measured data)")
    ap.add_argument("--epochs", type=int, default=10000, help="10000 in the repository")
    ap.add_argument("--stored", action="store_true",
                    help="no training: use hspice_ML/rram.va of the repository (the model trained by its authors) as it is")
    ap.add_argument("--threads", type=int, default=10)
    ap.add_argument("--seed", type=int, default=1, help="1 in the repository")
    ap.add_argument("--hspice", default="hspice")
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--suffix", default="")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    root = os.path.abspath(os.path.expanduser(a.root)); mr = os.path.abspath(os.path.expanduser(a.mr_root))
    for f in (os.path.join(root, "Codeupload", "Real RRAM", "input data", "IV_RRAM_TriD_35_1_real_new2.csv"),
              os.path.join(mr, "HSPICE_test", "1R.sp"), os.path.join(mr, "data", "test_RanD_real_new2.csv")):
        if not os.path.exists(f): sys.exit("not found: " + f)
    out = os.path.abspath(os.path.expanduser(a.out)) if a.out else os.path.join(mr, "results", "rerun", "baseline.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    tag = "baseline_lin_e%d%s%s" % (a.epochs, "" if a.seed == 1 else "_seed%d" % a.seed, a.suffix)
    if a.stored: tag = "baseline_lin_stored" + a.suffix
    run_dir = os.path.join(mr, "runs", tag)
    os.makedirs(os.path.join(run_dir, "verilog"), exist_ok=True)
    if a.stored:
        print("==== %s : model stored in the repository (no training)" % tag, flush=True)
        net, va = load_stored(root)
        shutil.copy(va, os.path.join(run_dir, "verilog", "rram.va"))
        info = dict(train_time=float("nan"), epochs=0, loss=float("nan"), n_meas=0, n_train=0, tf="")
    else:
        print("==== %s : training (%d epochs)" % (tag, a.epochs), flush=True)
        net, info = train(root, a.epochs, a.threads, a.seed)
        print("  training time %.1f s, %d epochs, loss %.5f" % (info["train_time"], info["epochs"], info["loss"]), flush=True)
        export_va(net, os.path.join(run_dir, "verilog", "rram.va"))
    np.savez(os.path.join(run_dir, "weights.npz"), *net["w"], scale=[net["v_min"], net["v_range"], net["y_min"], net["y_range"]])
    sims, rts = {}, {}
    for d, meas in (("HSPICE_train", "train_TriD_real_new2.csv"), ("HSPICE_test", "test_RanD_real_new2.csv")):
        deck_dir = os.path.join(run_dir, d); os.makedirs(os.path.join(deck_dir, "output"), exist_ok=True)
        make_deck(os.path.join(mr, d, "1R.sp"), os.path.join(deck_dir, "1R.sp"))
        emu = emulate(net, os.path.join(deck_dir, "1R.sp"))
        if a.dry_run:
            sims[d] = emu; rts[d] = float("nan")
        else:
            r = []
            for _ in range(a.repeat):
                with open(os.path.join(deck_dir, "output", "hspice_stdout.txt"), "w") as log:
                    ret = subprocess.call(a.hspice.split() + ["-i", "1R.sp", "-o", "output/output"], cwd=deck_dir, stdout=log, stderr=subprocess.STDOUT)
                try:
                    sims[d], rt = read_lis(os.path.join(deck_dir, "output", "output.lis"))
                except (RuntimeError, FileNotFoundError):
                    diagnose(deck_dir, ret); sys.exit(1)
                r.append(rt)
            rts[d] = float(np.mean(r))
        sims[d].to_csv(os.path.join(deck_dir, "output", "sim_data.csv"), index=False)
        emu.to_csv(os.path.join(deck_dir, "output", "emulation.csv"), index=False)
    e_tr = errors(os.path.join(mr, "data", "train_TriD_real_new2.csv"), sims["HSPICE_train"])
    e_te = errors(os.path.join(mr, "data", "test_RanD_real_new2.csv"), sims["HSPICE_test"])
    fields = ["run", "seed", "sim", "host", "threads", "tensorflow", "epochs", "loss", "n_meas", "n_train", "train_time",
              "runtime_train", "runtime_test", "rmse_train_mA", "rmsle_train_x1000", "rmse_test_mA", "rmsle_test_x1000",
              "rows_meas_test", "rows_sim_test"]
    row = dict(run=tag, seed=a.seed, sim="emulation" if a.dry_run else "hspice", host=socket.gethostname(), threads=a.threads,
               tensorflow=info["tf"], epochs=info["epochs"], loss=info["loss"], n_meas=info["n_meas"], n_train=info["n_train"],
               train_time=info["train_time"], runtime_train=rts["HSPICE_train"], runtime_test=rts["HSPICE_test"],
               rmse_train_mA=e_tr[0], rmsle_train_x1000=e_tr[1], rmse_test_mA=e_te[0], rmsle_test_x1000=e_te[1],
               rows_meas_test=e_te[2], rows_sim_test=e_te[3])
    new = not os.path.exists(out)
    with open(out, "a", newline="") as fo:
        wr = csv.DictWriter(fo, fieldnames=fields)
        if new: wr.writeheader()
        wr.writerow(row)
    print("  train %.1f s | runtime test %s s | train RMSE %.4f mA RMSLE %.4f | test RMSE %.4f mA RMSLE %.4f"
          % (row["train_time"], ("%.2f" % row["runtime_test"]) if not a.dry_run else "-", e_tr[0], e_tr[1], e_te[0], e_te[1]))
    print("results: " + out)


if __name__ == "__main__":
    main()
