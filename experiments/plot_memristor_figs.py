#!/usr/bin/env python3
"""Figures of the simulation results (style of utils/plot_MLmodel.py).

  python3 experiments/plot_memristor_figs.py --out figs [--sgpr RBF_Z50_neg] [--base baseline_lin_stored]

For train and test: T-V.pdf, T-I.pdf, I-V2D.pdf, I-V-S3D.pdf  (black: measured data, blue: conventional method, red: SGPR)
The state of the conventional method is taken from emulation.csv because it is not printed by HSPICE.
"""
import argparse, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SC = 1e3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mr-root", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); ap.add_argument("--out", default="figs")
    ap.add_argument("--sgpr", default="RBF_Z50_neg"); ap.add_argument("--base", default="baseline_lin_stored")
    ap.add_argument("--base-label", default="MLP")
    ap.add_argument("--sgpr-dir", default=None, help="directory with <run>_HSPICE_train.csv / _test.csv (results/rerun/data) instead of runs/")
    ap.add_argument("--base-source", default="sim_data.csv")
    a = ap.parse_args()
    mr = os.path.abspath(os.path.expanduser(a.mr_root)); os.makedirs(a.out, exist_ok=True)
    plt.rcParams.update({"font.size": 15})
    for kind, meas, d in (("train", "train_TriD_real_new2.csv", "HSPICE_train"), ("test", "test_RanD_real_new2.csv", "HSPICE_test")):
        m = pd.read_csv(os.path.join(mr, "data", meas))
        s = pd.read_csv(os.path.join(a.sgpr_dir, "%s_%s.csv" % (a.sgpr, d)) if a.sgpr_dir else os.path.join(mr, "runs", a.sgpr, d, "output", "sim_data.csv"))
        bdir = os.path.join(mr, "runs", a.base, d, "output")
        b = pd.read_csv(os.path.join(bdir, a.base_source)); be = pd.read_csv(os.path.join(bdir, "emulation.csv"))
        zlab = "SGPR(%d)" % int(a.sgpr.split("_Z")[1].split("_")[0]) if "_Z" in a.sgpr else "SGPR"
        lab = kind + " data"

        fig, ax = plt.subplots(figsize=(8, 4), tight_layout=True)
        ax.scatter(m["Time"], m["Voltage"], s=40, marker=".", color="black")
        ax.set_xlabel("Time[nsec]"); ax.set_ylabel("Voltage[V]"); ax.grid(linestyle="dotted", alpha=0.6)
        fig.savefig(os.path.join(a.out, "%s_T-V.pdf" % kind)); plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 4), tight_layout=True)
        ax.plot(s["Time"], s["Current"] * SC, color="red", lw=1, label=zlab, zorder=4)
        ax.plot(b["Time"], b["Current"] * SC, color="blue", lw=2.5, label=a.base_label, zorder=3)
        ax.scatter(m["Time"], m["Current"] * SC, s=40, marker=".", color="black", label=lab, zorder=1)
        ax.legend(frameon=False); ax.set_xlabel("Time[nsec]"); ax.set_ylabel("Current[mA]"); ax.grid(linestyle="dotted", alpha=0.6)
        fig.savefig(os.path.join(a.out, "%s_T-I.pdf" % kind)); plt.close(fig)

        fig, ax = plt.subplots(tight_layout=True)
        ax.plot(s["Voltage"], s["Current"] * SC, color="red", lw=1, label=zlab, zorder=4)
        ax.plot(b["Voltage"], b["Current"] * SC, color="blue", lw=2.5, label=a.base_label, zorder=3)
        ax.scatter(m["Voltage"], m["Current"] * SC, s=40, marker=".", color="black", label=lab, zorder=1)
        ax.legend(frameon=False); ax.set_xlabel("Voltage[V]"); ax.set_ylabel("Current[mA]"); ax.grid(linestyle="dotted", alpha=0.6)
        fig.savefig(os.path.join(a.out, "%s_I-V2D.pdf" % kind)); plt.close(fig)

        fig = plt.figure(tight_layout=True); ax = fig.add_subplot(projection="3d")
        n = min(len(be), len(b))
        ax.plot(s["S"], s["Voltage"], s["Current"] * SC, color="red", lw=1, label=zlab, zorder=6)
        ax.plot(be["S"][:n], b["Voltage"][:n], b["Current"][:n] * SC, color="blue", lw=2.5, label=a.base_label, zorder=5)
        ax.scatter(m["S"], m["Voltage"], m["Current"] * SC, s=60, marker=".", color="black", label=lab, zorder=1)
        ax.set_xlabel("State", labelpad=10); ax.set_ylabel("Voltage[V]", labelpad=10); ax.set_zlabel("Current[mA]", labelpad=10)
        ax.legend(frameon=False, bbox_to_anchor=(0.45, 1.10), loc="upper left")
        fig.savefig(os.path.join(a.out, "%s_I-V-S3D.pdf" % kind)); plt.close(fig)
    print("written to " + a.out)


if __name__ == "__main__":
    main()
