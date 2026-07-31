"""
Pareto front visualization.

Usage:
    python visualize.py                                  # default cascode run
    python visualize.py --pareto runs/pareto_common_gate.pkl
    python visualize.py --pareto runs/pareto_cascode.pkl --save front.png
"""

import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np


def plot_pareto(pareto, save_path: str = None):
    objs, params = pareto.to_arrays()
    if len(objs) == 0:
        print("No Pareto points to plot.")
        return

    nf_vals   = -objs[:, 0]   # back to NF in dB (lower=better)
    s21_vals  =  objs[:, 1]
    iip3_vals =  objs[:, 2]
    pdc_vals  = -objs[:, 3]

    fig = plt.figure(figsize=(14, 10))
    fig.suptitle("LNA RL Sizer: Pareto Front", fontsize=14, fontweight="bold")
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

    # panel 1: NF vs gain
    ax1 = fig.add_subplot(gs[0, 0])
    sc1 = ax1.scatter(nf_vals, s21_vals, c=pdc_vals, cmap="viridis_r",
                      s=60, edgecolors="k", linewidths=0.4, zorder=3)
    ax1.axvline(2.5, color="red", ls="--", lw=1, label="NF target 2.5 dB")
    ax1.axhline(15.0, color="blue", ls="--", lw=1, label="Gain target 15 dB")
    ax1.set_xlabel("NF (dB)")
    ax1.set_ylabel("S21 (dB)")
    ax1.set_title("NF vs Gain  [color = Pdc]")
    ax1.legend(fontsize=7)
    plt.colorbar(sc1, ax=ax1, label="Pdc (mW)")

    # panel 2: NF vs IIP3
    ax2 = fig.add_subplot(gs[0, 1])
    sc2 = ax2.scatter(nf_vals, iip3_vals, c=s21_vals, cmap="plasma",
                      s=60, edgecolors="k", linewidths=0.4, zorder=3)
    ax2.axvline(2.5, color="red", ls="--", lw=1, label="NF target")
    ax2.axhline(-5.0, color="green", ls="--", lw=1, label="IIP3 target -5 dBm")
    ax2.set_xlabel("NF (dB)")
    ax2.set_ylabel("IIP3 (dBm)")
    ax2.set_title("NF vs IIP3  [color = S21]")
    ax2.legend(fontsize=7)
    plt.colorbar(sc2, ax=ax2, label="S21 (dB)")

    # panel 3: power vs NF
    ax3 = fig.add_subplot(gs[1, 0])
    sc3 = ax3.scatter(pdc_vals, nf_vals, c=iip3_vals, cmap="coolwarm",
                      s=60, edgecolors="k", linewidths=0.4, zorder=3)
    ax3.axhline(2.5, color="red", ls="--", lw=1, label="NF target")
    ax3.axvline(20.0, color="orange", ls="--", lw=1, label="Pdc budget 20 mW")
    ax3.set_xlabel("Pdc (mW)")
    ax3.set_ylabel("NF (dB)")
    ax3.set_title("Power vs NF  [color = IIP3]")
    ax3.legend(fontsize=7)
    plt.colorbar(sc3, ax=ax3, label="IIP3 (dBm)")

    # panel 4: design-variable heatmap
    ax4 = fig.add_subplot(gs[1, 1])
    # use whatever design variables the stored points actually have
    # (they differ per topology)
    param_keys = [k for k in params[0].keys() if params[0].get(k) is not None]
    param_matrix = []
    for p in params:
        row = [p.get(k, 0.0) for k in param_keys]
        param_matrix.append(row)
    pm = np.array(param_matrix)
    # Normalize each column to [0,1]
    pm_norm = (pm - pm.min(axis=0)) / (np.ptp(pm, axis=0) + 1e-9)
    im = ax4.imshow(pm_norm.T, aspect="auto", cmap="YlOrRd",
                    interpolation="nearest")
    ax4.set_yticks(range(len(param_keys)))
    ax4.set_yticklabels(param_keys, fontsize=8)
    ax4.set_xlabel("Pareto point index")
    ax4.set_title("Design variables (normalized)")
    plt.colorbar(im, ax=ax4, label="Normalized value")

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to {save_path}")
    else:
        plt.tight_layout()
        plt.show()


def plot_training_progress(log_csv: str):
    """Plot reward curve from SB3 Monitor CSV."""
    import pandas as pd
    try:
        df = pd.read_csv(log_csv, skiprows=1)
    except Exception as e:
        print(f"Could not read log: {e}")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Training Progress")

    axes[0].plot(df["t"], df["r"], alpha=0.4, color="steelblue")
    # Rolling mean
    roll = df["r"].rolling(50, min_periods=1).mean()
    axes[0].plot(df["t"], roll, color="navy", lw=2)
    axes[0].set_xlabel("Timestep")
    axes[0].set_ylabel("Episode reward")
    axes[0].set_title("Episode reward over training")

    axes[1].plot(df["t"], df["l"], alpha=0.4, color="coral")
    roll_l = df["l"].rolling(50, min_periods=1).mean()
    axes[1].plot(df["t"], roll_l, color="darkred", lw=2)
    axes[1].set_xlabel("Timestep")
    axes[1].set_ylabel("Episode length")
    axes[1].set_title("Episode length over training")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pareto", type=str, default="runs/pareto.pkl",
                        help="Pareto pickle (train.py saves runs/pareto_<topology>.pkl)")
    parser.add_argument("--log",    type=str, default=None)
    parser.add_argument("--save",   type=str, default=None)
    args = parser.parse_args()

    # Fall back to the cascode file from the new naming scheme
    if not Path(args.pareto).exists() and args.pareto == "runs/pareto.pkl":
        alt = "runs/pareto_cascode.pkl"
        if Path(alt).exists():
            args.pareto = alt

    if Path(args.pareto).exists():
        with open(args.pareto, "rb") as f:
            pareto = pickle.load(f)
        plot_pareto(pareto, save_path=args.save)
    else:
        print(f"No Pareto file at {args.pareto}, run train.py first.")

    if args.log and Path(args.log).exists():
        plot_training_progress(args.log)
