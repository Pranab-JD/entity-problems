"""
Created on Sat Apr 18 06:06 2026

@author: Pranab JD, ChatGPT

Usage:

    input="/scratch/project_465002528/pjd/flux_tubes/tubes/fields/"
    output="/scratch/project_465002528/pjd/flux_tubes/tubes/plots/"

    python3 ../postprocessing/Plot_Bx_Bz_ycut.py "$input" "$output"

"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
from adios2 import Stream
import argparse, os, glob
import matplotlib.pyplot as plt

# ============================================================
# Args
# ============================================================
parser = argparse.ArgumentParser()
parser.add_argument("base", type=str, help="Directory with fields.*.bp")
parser.add_argument("outdir", type=str, help="Output directory")
parser.add_argument("--B0", type=float, default=1.0, help="Normalising magnetic field")

args = parser.parse_args()
base = args.base
outdir = args.outdir
B0 = args.B0

os.makedirs(outdir, exist_ok=True)

# ============================================================
# Find first file only
# ============================================================
files = sorted(glob.glob(f"{base}/fields.*.bp"))

if len(files) == 0:
    raise FileNotFoundError(f"No fields.*.bp files found in {base}")

fname = files[0]
step_str = fname.split(".")[-2]

print(f"Using first file only: {fname}", flush=True)

# ============================================================
# Read first dataset
# ============================================================
with Stream(fname, "r") as s:
    next(s.steps())

    x = np.asarray(s.read("X1"))
    y = np.asarray(s.read("X2"))
    z = np.asarray(s.read("X3"))

    Bx = np.asarray(s.read("fB1"))
    Bz = np.asarray(s.read("fB3"))

# ============================================================
# Indices
# ADIOS field shape is assumed to be (z, y, x)
# ============================================================
kz_mid = Bx.shape[0] // 2
ix_mid = Bx.shape[2] // 2

# ============================================================
# 1D y-cuts at fixed x-midplane and z-midplane
# ============================================================
Bx_ycut = Bx[kz_mid, :, ix_mid] / B0
Bz_ycut = Bz[kz_mid, :, ix_mid] / B0

# ============================================================
# Plot
# ============================================================
fig, ax1 = plt.subplots(figsize=(9, 4.8))

ax2 = ax1.twinx()

line1, = ax1.plot(y, Bx_ycut, "r-", lw=2.0, label=r"$B_x/B_0$")
line2, = ax2.plot(y, Bz_ycut, "b-", lw=2.0, label=r"$B_z/B_0$")

ax1.axvline(y[len(y) // 2], color="k", ls="--", lw=1.2)

ax1.set_xlabel(r"$y/d_{e0}$", fontsize=16)
ax1.set_ylabel(r"$B_x/B_0$", fontsize=16, color="red")
ax2.set_ylabel(r"$B_z/B_0$", fontsize=16, color="blue")

ax1.tick_params(axis="y", labelcolor="red", labelsize=12, length=6)
ax2.tick_params(axis="y", labelcolor="blue", labelsize=12, length=6)
ax1.tick_params(axis="x", labelsize=12, length=6)

ax1.set_xlim(y.min(), y.max())

ax1.grid(False)
ax2.grid(False)

fig.tight_layout()

outfile = f"{outdir}/Bx_Bz_ycut_{step_str}.png"
fig.savefig(outfile, dpi=200)
plt.close(fig)

print(f"Saved {outfile}", flush=True)
print("\nDone.", flush=True)