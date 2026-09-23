"""
Created on Thu Sep 10 2026

@author: Pranab JD, Claude AI

Plot magnetic-component and kinetic energy histories from Entity's *_stats.csv
for the relativistic Harris-sheet run. All four panels are normalised to the
INITIAL in-plane magnetic energy B_x^2(t=0)

Usage
-----
    csv="/scratch/project_465002528/pjd/RMR_ie_2D/RMR_stats.csv"
    out="/scratch/project_465002528/pjd/RMR_ie_2D/plots"

    python3 -u Energy_history.py "$csv" "$out"
    python3 -u Energy_history.py "$csv" "$out" --list_cols   # show CSV headers

    Optional:
        --fields  directory with fields.*.bp files used to get L_x
                  (default: strip "_stats.csv" from the CSV path to recover
                   <simulation.name>, then use <simulation.name>/fields)
        --Lx      box length along X in code units; overrides --fields
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import argparse, os, glob
import matplotlib.pyplot as plt

#! ============================================================
#! USER SETTINGS
#! ============================================================
#! L_x definition:
#!   False -> X1.max() - X1.min()   (cell centres; identical to Plot_Bx_Jz_rho.py)
#!   True  -> X1e.max() - X1e.min() (cell edges; exact box length)
USE_CELL_EDGES = False

#! ============================================================
#! Args
#! ============================================================
parser = argparse.ArgumentParser()
parser.add_argument("csv",    type=str, help="Path to the *_stats.csv file")
parser.add_argument("outdir", type=str, help="Output directory for the PNG")
parser.add_argument("--fields", type=str, default=None,
                    help="Directory with fields.*.bp files for L_x "
                         "(default: <directory of the CSV>/fields)")
parser.add_argument("--Lx", type=float, default=None,
                    help="Box length along X (code units); overrides --fields")
parser.add_argument("--list_cols", action="store_true",
                    help="Print the CSV column names and exit")
args = parser.parse_args()

CSV    = args.csv
OUTDIR = args.outdir
TCOL   = "time"   #! time column name

#! ------------------------------------------------------------
#! Default fields dir: recover the run directory from the CSV name.
#!   Entity writes stats to  <simulation.name>_stats.csv  and field
#!   output to  <simulation.name>/fields/  (see metadomain_stats.cpp).
#!   So stripping the "_stats.csv" suffix from the CSV path gives back
#!   <simulation.name>, i.e. the run directory that holds fields/.
#!   e.g.  .../RMR_stats.csv        -> .../RMR/fields
#!         .../RMR/_stats.csv       -> .../RMR/fields   (trailing-slash name)
#!   Fallback (CSV not named *_stats.csv): fields/ next to the CSV.
#! ------------------------------------------------------------
def default_fields_dir(csv_path):
    p = os.path.abspath(csv_path)
    if p.endswith("_stats.csv"):
        simname = p[: -len("_stats.csv")]          #! = simulation.name
        return os.path.join(simname, "fields")
    return os.path.join(os.path.dirname(p), "fields")

FIELDS_DIR = args.fields if args.fields is not None else default_fields_dir(CSV)

os.makedirs(OUTDIR, exist_ok=True)

#! ============================================================
#! Read CSV (comma-separated, possible trailing comma per row)
#! ============================================================
with open(CSV, "r") as f:
    header_line = f.readline()
colnames = [c.strip() for c in header_line.strip().split(",")]
colnames = [c for c in colnames if c != ""]

if args.list_cols:
    print(f"\nColumns in {os.path.basename(CSV)}:")
    for c in colnames:
        print("   ", c)
    raise SystemExit(0)

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
        f"  Run with --list_cols and edit the candidate names below.")

t    = find(TCOL, "Time", "t")
B1sq = find("B1^2", "Bx^2", "B_1^2", "B1_2")
B2sq = find("B2^2", "By^2", "B_2^2", "B2_2")
B3sq = find("B3^2", "Bz^2", "B_3^2", "B3_2")
T00  = find("T00", "T_00", "Ttt")
Rho  = find("Rho", "rho", "N")

#! ============================================================
#! L_x: box length along X, same definition as Plot_Bx_Jz_rho.py
#!   priority: --Lx  >  X1 (or X1e) from the first fields.*.bp file
#! ============================================================
def get_Lx():
    if args.Lx is not None:
        return float(args.Lx), "--Lx (command line)"

    ffiles = sorted(glob.glob(f"{FIELDS_DIR}/fields.*.bp"))
    if len(ffiles) == 0:
        raise SystemExit(
            f"Cannot determine L_x: no fields.*.bp in {FIELDS_DIR}\n"
            f"  - pass --fields <dir> or --Lx <value>")

    from adios2 import Stream              #! imported only when needed
    var = "X1e" if USE_CELL_EDGES else "X1"
    with Stream(ffiles[0], "r") as s:
        next(s.steps())
        x = np.asarray(s.read(var))        #! global X coordinates (1D)
    return float(x.max() - x.min()), f"{var} in {os.path.basename(ffiles[0])}"

LX, LX_SRC = get_Lx()
if not (LX > 0):
    raise SystemExit(f"Invalid L_x = {LX} (from {LX_SRC})")
print(f"L_x = {LX:.6g}  (from {LX_SRC})", flush=True)

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
KE_f   = KE   / norm0

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