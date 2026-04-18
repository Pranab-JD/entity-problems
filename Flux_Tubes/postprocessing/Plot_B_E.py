#!/usr/bin/env python3

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from adios2 import Stream
import argparse, os, glob

# ============================================================
# args
# ============================================================
parser = argparse.ArgumentParser()
parser.add_argument("base", type=str, help="Directory with fields.*.bp")
parser.add_argument("outdir", type=str, help="Output directory")

args   = parser.parse_args()
base   = args.base
outdir = args.outdir

os.makedirs(outdir, exist_ok=True)

# ============================================================
# find all files
# ============================================================
files = sorted(glob.glob(f"{base}/fields.*.bp"))

print(f"Found {len(files)} files")

# ============================================================
# loop over ALL files (no stride)
# ============================================================
for fname in files:

    step_str = fname.split(".")[-2]
    print(f"Processing {fname}")

    with Stream(fname, "r") as s:
        next(s.steps())

        # coords
        x = np.asarray(s.read("X1"))
        y = np.asarray(s.read("X2"))
        z = np.asarray(s.read("X3"))

        # fields (3D)
        Bx = np.asarray(s.read("fB1"))
        By = np.asarray(s.read("fB2"))
        Bz = np.asarray(s.read("fB3"))

        Ex = np.asarray(s.read("fE1"))
        Ey = np.asarray(s.read("fE2"))
        Ez = np.asarray(s.read("fE3"))

        # ----------------------------------------------------
        # slice (mid-plane z)
        # ----------------------------------------------------
        k = Bx.shape[2] // 2

        Bx = Bx[:, :, k]
        By = By[:, :, k]
        Bz = Bz[:, :, k]

        Ex = Ex[:, :, k]
        Ey = Ey[:, :, k]
        Ez = Ez[:, :, k]

    # ========================================================
    # plotting
    # ========================================================
    fig, axs = plt.subplots(2, 3, figsize=(14, 6))

    fields = [(Bx, "Bx"), (By, "By"), (Bz, "Bz"),
              (Ex, "Ex"), (Ey, "Ey"), (Ez, "Ez")]

    for ax, (data, title) in zip(axs.flat, fields):
        im = ax.imshow(data, origin="lower", aspect="auto",
                        extent=[x.min(), x.max(), y.min(), y.max()], cmap="seismic")

        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()

    outfile = f"{outdir}/fields_{step_str}.png"
    fig.savefig(outfile, dpi=150)
    plt.close(fig)

    print(f"Saved {outfile}")

print("Done.")
