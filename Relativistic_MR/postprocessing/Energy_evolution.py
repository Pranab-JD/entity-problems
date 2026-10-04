"""
Created on Thu Sep 10 2026

@author: Pranab JD, Claude AI

Plot magnetic and kinetic energy histories from entity's *_stats.csv
All four panels are normalised to the INITIAL in-plane magnetic energy B_x^2(t=0)

Usage
-----
    sfolder="/scratch/project_465003132/RMR_pair_3D/sigma_10_Z100/"
    folder="/scratch/project_465003132/RMR_pair_3D/sigma_10_Z100/RMR/"
    output="${folder}/plots"

    Lx=200.0; larmor=0.3162

    srun -N 1 -n 1 python3 -u ../postprocessing/Energy_evolution.py "${sfolder}RMR_stats.csv" "$output" \
    --Lx "$Lx" --larmor0 "$larmor"

"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import argparse, os
import matplotlib.pyplot as plt

#! ============================================================
#! Args
#! ============================================================
parser = argparse.ArgumentParser()
parser.add_argument("csv",          type=str)
parser.add_argument("outdir",       type=str)
parser.add_argument("--Lx",         type=float, required=True)
parser.add_argument("--larmor0",    type=float, default=None)
parser.add_argument("--skindepth0", type=float, default=1.0)

args = parser.parse_args()

CSV    = args.csv
OUTDIR = args.outdir
TCOL   = "time"   #! time column name

os.makedirs(OUTDIR, exist_ok=True)

#! ============================================================
#! Read CSV (comma-separated, possible trailing comma per row)
#! ============================================================
with open(CSV, "r") as f:
    header_line = f.readline()
colnames = [c.strip() for c in header_line.strip().split(",")]
colnames = [c for c in colnames if c != ""]

data = np.genfromtxt(CSV, delimiter=",", skip_header=1)
if data.ndim == 1:
    data = data[None, :]
if data.shape[1] == len(colnames) + 1 and np.all(np.isnan(data[:, -1])):
    data = data[:, :-1]

col = {name: data[:, i] for i, name in enumerate(colnames)}

#! ============================================================
#! Column lookup tolerant of naming variants
#! ============================================================
def find(*candidates):
    for name in candidates:
        if name in col:
            return col[name]
    raise SystemExit(
        f"None of {candidates} found in CSV.\n"
        f"  Available: {list(col.keys())}\n"
        f"  Edit the candidate names below.")

t    = find(TCOL, "Time", "t")
B1sq = find("B1^2", "Bx^2", "B_1^2", "B1_2")
B2sq = find("B2^2", "By^2", "B_2^2", "B2_2")
B3sq = find("B3^2", "Bz^2", "B_3^2", "B3_2")
T00  = find("T00", "T_00", "Ttt")
Rho  = find("Rho", "rho", "N")

#! ============================================================
#! L_x: box length along X, in code units
#! ============================================================
LX = float(args.Lx)
if not (LX > 0):
    raise SystemExit(f"Invalid L_x = {LX}")
print(f"L_x = {LX:.6g}", flush=True)

t_lc = t / LX                              #! light-crossing times of L_x (c = 1)

#! ============================================================
#! Energies and normalisation to INITIAL B_x^2
#! ============================================================
E_Bx = 0.5 * B1sq
E_By = 0.5 * B2sq
E_Bz = 0.5 * B3sq
KE   = T00 - Rho

norm0 = E_Bx[0]
if norm0 == 0 or not np.isfinite(norm0):
    raise SystemExit(f"Initial B_x^2 energy at row {0} is {norm0}; cannot normalise.")

E_Bx_f = E_Bx / norm0
E_By_f = E_By / norm0
E_Bz_f = E_Bz / norm0
sigma0 = (args.skindepth0 / args.larmor0) ** 2
KE_f   = KE / (sigma0 * norm0)

#! ============================================================
#! Plot 2x2
#! ============================================================
fig, axs = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)

axs[0, 0].plot(t_lc, E_Bx_f, color="blue",  lw=1.8)
axs[0, 0].set_ylabel(r"$B_x^2(t) / B_x^2(t=0)$", fontsize=14)

axs[0, 1].plot(t_lc, E_By_f, color="green", lw=1.8)
axs[0, 1].set_ylabel(r"$B_y^2(t) / B_x^2(t=0)$", fontsize=14)

axs[1, 0].plot(t_lc, E_Bz_f, color="purple", lw=1.8)
axs[1, 0].set_ylabel(r"$B_z^2(t) / B_x^2(t=0)$", fontsize=14)

axs[1, 1].plot(t_lc, KE_f, color="red", lw=1.8)
axs[1, 1].set_ylabel(r"$KE(t) / B_x^2(t=0)$", fontsize=14)

for ax in axs.flat:
    ax.set_xlabel(r"$t\,c/L_x$", fontsize=14)
    ax.tick_params(axis="both", labelsize=12)

outfile = os.path.join(OUTDIR, "energy_history.png")
fig.savefig(outfile, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nSaved {outfile}", flush=True)