"""
Created on Sat Jun 13 2026

@author: Pranab JD, Claude AI

Plot particle spectra dN/dgamma from spectra.*.bp files (relativistic Harris
sheet Entity run), overlaid on a log-log plot with colour encoding time in
LIGHT-CROSSING TIMES of the box along X

Usage
-----
    spectra="/scratch/project_465002528/pjd/RMR_2D/spectra"
    output="/scratch/project_465002528/pjd/RMR_2D/plots"

    srun python3 -u Plot_spectra.py "$spectra" "$output"            # ion-electron
    srun python3 -u Plot_spectra.py "$spectra" "$output" --pair     # pair plasma

    Optional:
        --fields     directory with fields.*.bp files used to get L_x
                     (default: <parent of spectra dir>/fields)
        --Lx         box length along X in code units; overrides --fields
        --species    comma-separated species indices to plot   (default "1,2")
        --pair       species 2 is a POSITRON (light X limits, "Positrons" title);
                     default treats species 2 as an ION (ion X limits, "Ions").
        --list_vars  print the BP variable names and exit      (schema discovery)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
from adios2 import Stream
import argparse, os, glob
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize

#! ============================================================
#! USER SETTINGS
#! ============================================================
#! Axis limits (set None for auto on that side).
#! Y limits: shared by ALL species.
Y_MIN = 1e3
Y_MAX = 6e7

#! X limits for the LIGHT species (electrons and positrons).
XLIGHT_MIN = 5e-6
XLIGHT_MAX = 4e3

#! X limits for the HEAVY species (ions).
XION_MIN = 5e-5
XION_MAX = 1e0

#! Larmor radius scale r_L0 = mc^2/(q B0), in d0 units.
#! CHECK: must match [scales] larmor0 of the run (pair TOML: 1.0, ion TOML: 5.44662e-4).
LARMOR0 = 0.2

#! L_x definition:
#!   False -> X1.max() - X1.min()   (cell centres; identical to Plot_Bx_Jz_rho.py)
#!   True  -> X1e.max() - X1e.min() (cell edges; exact box length)
USE_CELL_EDGES = False

#! ============================================================
#! Schema settings — EDIT if --list_vars shows different names
#! ============================================================
#! Energy/gamma bin array (first match wins). Entity writes 'sEbn'.
BIN_VAR_CANDIDATES  = ["sEbn", "ebins", "e_bins", "bins", "sebn", "energy", "gamma"]
#! Per-species count arrays ({s} -> species index). Entity writes 'sN_1','sN_2'.
COUNT_VAR_TEMPLATES = ["sN_{s}", "sN{s}", "spectrum_{s}", "f_{s}", "N{s}"]

#! If counts are raw dN per bin and you want dN/dE, divide by bin width.
NORMALISE_BY_BIN_WIDTH = False

TIME_KEY = "Time"

#! ============================================================
#! Args
#! ============================================================
parser = argparse.ArgumentParser()
parser.add_argument("base",   type=str, help="Directory with spectra.*.bp files")
parser.add_argument("outdir", type=str, help="Output directory for the PNG")
parser.add_argument("--fields", type=str, default=None,
                    help="Directory with fields.*.bp files for L_x "
                         "(default: <parent of spectra dir>/fields)")
parser.add_argument("--Lx", type=float, default=None,
                    help="Box length along X (code units); overrides --fields")
parser.add_argument("--species", type=str, default="1,2",
                    help="Comma-separated species indices (default '1,2')")
parser.add_argument("--pair", action="store_true",
                    help="Species 2 is a positron (light X limits); default: ion")
parser.add_argument("--list_vars", action="store_true",
                    help="Print BP variable names from the first file and exit")
args = parser.parse_args()

BASE    = args.base
OUTDIR  = args.outdir
SPECIES = [int(s) for s in args.species.split(",") if s.strip() != ""]
IS_PAIR = args.pair

#! default fields dir: sibling of the spectra dir (…/run/spectra -> …/run/fields)
FIELDS_DIR = args.fields if args.fields is not None else \
             os.path.join(os.path.dirname(os.path.normpath(BASE)), "fields")

os.makedirs(OUTDIR, exist_ok=True)

#! ============================================================
#! Species type / label / X-limits keyed by species INDEX
#!   species 1  -> electron (light)
#!   species 2  -> positron (light) if --pair, else ion (heavy)
#!   other      -> treated as light by default
#! ============================================================
def species_info(sp):
    if sp == 1:
        return "Electron", XLIGHT_MIN, XLIGHT_MAX
    if sp == 2:
        if IS_PAIR:
            return "Positron", XLIGHT_MIN, XLIGHT_MAX
        return "Ions", XION_MIN, XION_MAX
    return f"Species {sp}", XLIGHT_MIN, XLIGHT_MAX

#! ============================================================
#! Find files
#! ============================================================
files = sorted(glob.glob(f"{BASE}/spectra.*.bp"))
if len(files) == 0:
    raise SystemExit(
        f"No spectra.*.bp files found in {BASE}\n"
        f"  - Is [output.spectra] enable = true in the TOML? (defaults to false)\n"
        f"  - Did the run write spectra to this directory?")

print(f"Found {len(files)} spectra files", flush=True)

#! ============================================================
#! Helpers
#! ============================================================
def step_from_fname(fname):
    stem = os.path.basename(fname).rsplit(".", 1)[0]
    try:
        return int(stem.split(".")[-1])
    except ValueError:
        return -1

def read_time(stream):
    #! simulation time in code units (variable first, then attribute)
    for getter in (lambda: stream.read(TIME_KEY),
                   lambda: stream.read_attribute(TIME_KEY)):
        try:
            v = getter()
            if v is not None:
                return float(np.asarray(v).ravel()[0])
        except Exception:
            pass
    return float("nan")

def available_vars(stream):
    try:
        return list(stream.available_variables().keys())
    except Exception:
        try:
            return list(stream.available_variables())
        except Exception:
            return []

def pick_bin_var(varnames):
    for cand in BIN_VAR_CANDIDATES:
        if cand in varnames:
            return cand
    return None

def pick_count_var(varnames, s):
    for tmpl in COUNT_VAR_TEMPLATES:
        name = tmpl.format(s=s)
        if name in varnames:
            return name
    return None

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

    var = "X1e" if USE_CELL_EDGES else "X1"
    with Stream(ffiles[0], "r") as s:
        next(s.steps())
        x = np.asarray(s.read(var))           #! global X coordinates (1D)
    return float(x.max() - x.min()), f"{var} in {os.path.basename(ffiles[0])}"

#! ============================================================
#! --list_vars : dump schema and exit
#! ============================================================
if args.list_vars:
    with Stream(files[0], "r") as s:
        next(s.steps())
        names = available_vars(s)
    print(f"\nVariables in {os.path.basename(files[0])}:")
    for n in names:
        print("   ", n)
    print("\nEdit BIN_VAR_CANDIDATES / COUNT_VAR_TEMPLATES at the top if the "
          "defaults don't match.\n")
    raise SystemExit(0)

LX, LX_SRC = get_Lx()
if not (LX > 0):
    raise SystemExit(f"Invalid L_x = {LX} (from {LX_SRC})")
print(f"L_x = {LX:.6g}  (from {LX_SRC})", flush=True)

#! ============================================================
#! Read all spectra
#!   spectra_by_species[s] = list of (t_lc, bins, counts)
#!   t_lc = t_code / L_x   (light-crossing times of L_x, c = 1)
#! ============================================================
spectra_by_species = {s: [] for s in SPECIES}
times_seen = []

for fname in files:
    with Stream(fname, "r") as s:
        next(s.steps())
        names  = available_vars(s)
        t_code = read_time(s)
        t_lc   = t_code / LX                  #! NaN propagates if Time is missing

        bin_var = pick_bin_var(names)
        if bin_var is None:
            #! auto-detect: a 1D monotonic-increasing float array is likely bins
            for n in names:
                try:
                    arr = np.asarray(s.read(n)).ravel()
                except Exception:
                    continue
                if arr.ndim == 1 and arr.size > 3 and np.all(np.diff(arr) > 0):
                    bin_var = n
                    break
        if bin_var is None:
            print(f"  [skip] {os.path.basename(fname)}: no bin array found", flush=True)
            continue

        bins = np.asarray(s.read(bin_var)).ravel()

        for sp in SPECIES:
            cvar = pick_count_var(names, sp)
            if cvar is None:
                continue
            counts = np.asarray(s.read(cvar)).ravel()
            spectra_by_species[sp].append((t_lc, bins, counts))

    times_seen.append(t_lc)

for sp in SPECIES:
    print(f"    species {sp}: {len(spectra_by_species[sp])} spectra", flush=True)

#! ============================================================
#! Plot — one panel per species, overlaid lines coloured by t c/L_x
#! ============================================================
finite_times = [t for t in times_seen if np.isfinite(t)]
tmin = min(finite_times) if finite_times else 0.0
tmax = max(finite_times) if finite_times else 1.0
norm = Normalize(vmin=tmin, vmax=tmax)
cmap = cm.jet

n_sp = len([sp for sp in SPECIES if len(spectra_by_species[sp]) > 0])
if n_sp == 0:
    raise SystemExit("No spectra read — run with --list_vars and fix the var names.")

fig, axs = plt.subplots(1, n_sp, figsize=(6 * n_sp, 5), squeeze=False, constrained_layout=True)
axs = axs[0]

panel = 0
for sp in SPECIES:
    series = spectra_by_species[sp]
    if len(series) == 0:
        continue
    ax = axs[panel]; panel += 1

    label, xlo, xhi = species_info(sp)

    for (t_lc, bins, counts) in series:
        #! bin centres: if bins are EDGES (len = counts+1), take midpoints
        if bins.size == counts.size + 1:
            x = 0.5 * (bins[1:] + bins[:-1])
            widths = np.diff(bins)
        else:
            x = bins
            widths = np.gradient(bins)

        y = counts.astype(float)
        if NORMALISE_BY_BIN_WIDTH:
            with np.errstate(divide="ignore", invalid="ignore"):
                y = np.where(widths > 0, y / widths, 0.0)

        colour = cmap(norm(t_lc)) if np.isfinite(t_lc) else "gray"
        good = (x > 0) & (y > 0)         #! mask non-positive for log-log
        ax.plot(x[good], y[good], color=colour, lw=1.2, alpha=0.85)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\gamma - 1$", fontsize=14)
    ax.set_ylabel(r"$dN/d\gamma$", fontsize=14)
    ax.set_title(f"{label} spectrum", fontsize=14)

    #! X limits by species type; Y limits shared. (None -> auto on that side.)
    ax.set_xlim(left=xlo, right=xhi)
    ax.set_ylim(bottom=Y_MIN, top=Y_MAX)

    #! Top x-axis: relativistic gyroradius corresponding to gamma-1.
    #! CHECK: valid for mass-1 species only; for ions r_L = mass_ratio * LARMOR0 * u.
    def gamma_minus_one_to_rL(gm1):
        gamma = gm1 + 1.0
        return LARMOR0 * np.sqrt(gamma**2 - 1.0)

    def rL_to_gamma_minus_one(rL):
        gamma = np.sqrt(1.0 + (rL / LARMOR0)**2)
        return gamma - 1.0

    secax = ax.secondary_xaxis(
        "top",
        functions=(gamma_minus_one_to_rL, rL_to_gamma_minus_one)
    )
    secax.set_xlabel(r"$r_L/d_0$", fontsize=14)

#! shared colourbar: light-crossing times of L_x
sm = cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])
cbar = fig.colorbar(sm, ax=list(axs[:panel]), fraction=0.046, pad=0.02)
cbar.set_label(r"$t\,c/L_x$", fontsize=14)

outfile = os.path.join(OUTDIR, "spectra.png")
fig.savefig(outfile, dpi=150, bbox_inches="tight")
plt.close(fig)

print(f"\nSaved {outfile}", flush=True)