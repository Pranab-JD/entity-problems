"""
Created on Sun May 10 2026

@author: Pranab JD, ChatGPT

Usage:

    input="/scratch/project_465002528/pjd/flux_tubes/"
    output="/scratch/project_465002528/pjd/flux_tubes/plots/" --Lx 500 --Ly 500 --Lz 500

    python3 ../postprocessing/Plot_energy.py "$input" "$output"

Input:

    /scratch/project_465002528/pjd/flux_tubes/tubes_stats.csv
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
parser.add_argument("base", type=str, help="Directory containing tubes_stats.csv")
parser.add_argument("outdir", type=str, help="Output directory")
parser.add_argument("--Lx", type=float, required=True, help="Domain length in x")
parser.add_argument("--Ly", type=float, required=True, help="Domain length in y")
parser.add_argument("--Lz", type=float, required=True, help="Domain length in z")
parser.add_argument("--no_half_factor", action="store_true", help="Use V * field^2 instead of 0.5 * V * field^2")

args = parser.parse_args()
base = args.base
outdir = args.outdir

os.makedirs(outdir, exist_ok=True)

#! ============================================================
#! Volume
#! ============================================================
volume = args.Lx * args.Ly * args.Lz

if args.no_half_factor:
    field_factor = 1.0
else:
    field_factor = 0.5

print(f"Using volume V = {volume}", flush=True)
print(f"Using field energy factor = {field_factor}", flush=True)

#! ============================================================
#! Read data
#! ============================================================
infile = f"{base}/tubes_stats.csv"

if not os.path.isfile(infile):
    raise FileNotFoundError(f"Could not find input file: {infile}")

print(f"Reading {infile}", flush=True)

data = np.genfromtxt(
    infile,
    delimiter=",",
    names=True,
    dtype=None,
    encoding=None,
    autostrip=True,
    deletechars=""
)

names = list(data.dtype.names)

print("Columns found:", names, flush=True)

required_columns = ["time", "B1^2", "B2^2", "B3^2", "E1^2", "E2^2", "E3^2", "Rho", "T00"]

for col in required_columns:
    if col not in names:
        raise KeyError(f"Missing column '{col}'. Available columns are: {names}")

time = data["time"]

B1_sq = data["B1^2"]
B2_sq = data["B2^2"]
B3_sq = data["B3^2"]

E1_sq = data["E1^2"]
E2_sq = data["E2^2"]
E3_sq = data["E3^2"]

Rho = data["Rho"]
T00 = data["T00"]

#! ============================================================
#! Convert mean densities to total energies
#! ============================================================
B_energy = field_factor * volume * (B1_sq + B2_sq + B3_sq)
E_energy = field_factor * volume * (E1_sq + E2_sq + E3_sq)
K_energy = volume * (T00 - Rho)

total_energy = B_energy + E_energy + K_energy

#! ============================================================
#! Plot
#! ============================================================
fig, ax = plt.subplots(1, 1, figsize=(8, 5), dpi=150)

ax.plot(time, B_energy, color="blue",  linewidth=2.0, label=r"Magnetic")
ax.plot(time, E_energy, color="green", linewidth=2.0, label=r"Electric")
ax.plot(time, K_energy, color="red",   linewidth=2.0, label=r"Kinetic")
# ax.plot(time, total_energy, color="black", linewidth=2.0, label=r"Total")

ax.set_xlabel(r"Time")
ax.set_ylabel(r"Energy")
ax.set_title("Energy evolution")

ax.tick_params(axis="both", which="major", labelsize=10, length=6)
ax.tick_params(axis="both", which="minor", labelsize=8, length=3)

ax.legend(frameon=False)

fig.tight_layout()

outfile = f"{outdir}/energy_evolution.png"
fig.savefig(outfile, dpi=300)
plt.close(fig)

print(f"Saved {outfile}", flush=True)
print("\nDone.", flush=True)