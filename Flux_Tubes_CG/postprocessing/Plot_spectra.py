"""
Created on Sat Jun 13 17:32 2026

@author: Pranab JD, Claude AI

Plot particle spectra dN/dgamma from spectra.*.bp files,
overlaid on a single log-log plot with colour encoding time.

Usage
-----
    folder="/scratch/project_465002528/pjd/flux_tubes_CG/2D_512_test/"
    spectra="${folder}tubes/spectra"
    output="${folder}tubes/plots"

    srun python3 -u ../postprocessing/Plot_spectra.py "$spectra" "$output" --species 1,2 \
                --r_j 128.0 --xmin 5e-5 --xmax 1e2 --ymin 5e3 --ymax 3e7

    Optional:
        --r_j        tube radius in d_e, for the t c/R label    (default 100.0)
        --species    comma-separated species indices to plot    (default "1,2")
        --xmin       lower x-axis (gamma-1) limit               (default: auto)
        --xmax       upper x-axis (gamma-1) limit               (default: auto)
        --ymin       lower y-axis (dN/dgamma) limit             (default: auto)
        --ymax       upper y-axis (dN/dgamma) limit             (default: auto)
        --list_vars  just print the BP variable names and exit  (schema discovery)

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
#! Schema settings — EDIT if --list_vars shows different names
#! ============================================================
# Candidate names for the energy/gamma bin array (first match wins).
# Entity writes 'sEbn' (confirmed via --list_vars).
BIN_VAR_CANDIDATES   = ["sEbn", "ebins", "e_bins", "bins", "sebn", "energy", "gamma"]
# Candidate templates for per-species count arrays ({s} -> species index).
# Entity writes 'sN_1', 'sN_2' (confirmed via --list_vars).
COUNT_VAR_TEMPLATES  = ["sN_{s}", "sN{s}", "spectrum_{s}", "f_{s}", "N{s}"]

# If counts are raw dN per bin and you want dN/dE, divide by bin width.
# If Entity already stores dN/dE, set this False.
NORMALISE_BY_BIN_WIDTH = False

TIME_KEY = "Time"

#! ============================================================
#! Args
#! ============================================================
parser = argparse.ArgumentParser()
parser.add_argument("base",   type=str, help="Directory with spectra.*.bp files")
parser.add_argument("outdir", type=str, help="Output directory for the PNG")
parser.add_argument("--r_j",  type=float, default=100.0,
                    help="Tube radius in d_e, for the t c/R label (default 100.0)")
parser.add_argument("--species", type=str, default="1,2",
                    help="Comma-separated species indices (default '1,2')")
parser.add_argument("--xmin", type=float, default=None,
                    help="Lower x-axis (gamma-1) limit (default: auto)")
parser.add_argument("--xmax", type=float, default=None,
                    help="Upper x-axis (gamma-1) limit (default: auto)")
parser.add_argument("--ymin", type=float, default=None,
                    help="Lower y-axis (dN/dgamma) limit (default: auto)")
parser.add_argument("--ymax", type=float, default=None,
                    help="Upper y-axis (dN/dgamma) limit (default: auto)")
parser.add_argument("--list_vars", action="store_true",
                    help="Print BP variable names from the first file and exit")
args = parser.parse_args()

BASE     = args.base
OUTDIR   = args.outdir
R_J      = args.r_j
SPECIES  = [int(s) for s in args.species.split(",") if s.strip() != ""]
XMIN     = args.xmin
XMAX     = args.xmax
YMIN     = args.ymin
YMAX     = args.ymax

os.makedirs(OUTDIR, exist_ok=True)

#! ============================================================
#! Find files
#! ============================================================
files = sorted(glob.glob(f"{BASE}/spectra.*.bp"))
if len(files) == 0:
    raise SystemExit(
        f"No spectra.*.bp files found in {BASE}\n"
        f"  - Is [output.spectra] enable = true in the TOML? (it defaults to false)\n"
        f"  - Did the run actually write spectra, and to this directory?\n"
        f"  (Refusing to fall back to *.bp — that would read fields.*.bp by mistake.)")

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
    """Return a list of variable names present in the BP step."""
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
#! --list_vars : dump schema and exit
#! ============================================================
if args.list_vars:
    with Stream(files[0], "r") as s:
        next(s.steps())
        names = available_vars(s)
    print(f"\nVariables in {os.path.basename(files[0])}:")
    for n in names:
        print("   ", n)
    print("\nEdit BIN_VAR_CANDIDATES / COUNT_VAR_TEMPLATES at the top of this "
          "script if the defaults don't match.\n")
    raise SystemExit(0)

#! ============================================================
#! Read all spectra
#!   spectra_by_species[s] = list of (time, bins, counts)
#! ============================================================
spectra_by_species = {s: [] for s in SPECIES}
times_seen = []

for fname in files:
    with Stream(fname, "r") as s:
        next(s.steps())
        names  = available_vars(s)
        t_code = read_time(s)

        bin_var = pick_bin_var(names)
        if bin_var is None:
            # auto-detect: a 1D, monotonic-increasing float array is likely bins
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
            spectra_by_species[sp].append((t_code, bins, counts))

    times_seen.append(t_code)

# report what we got
for sp in SPECIES:
    print(f"    species {sp}: {len(spectra_by_species[sp])} spectra", flush=True)

#! ============================================================
#! Plot — one panel per species, overlaid lines coloured by time
#! ============================================================
finite_times = [t for t in times_seen if np.isfinite(t)]
tmin = min(finite_times) if finite_times else 0.0
tmax = max(finite_times) if finite_times else 1.0
norm = Normalize(vmin=tmin, vmax=tmax)
cmap = cm.jet

n_sp = len([sp for sp in SPECIES if len(spectra_by_species[sp]) > 0])
if n_sp == 0:
    raise SystemExit("No spectra read — run with --list_vars and fix the var names.")

fig, axs = plt.subplots(1, n_sp, figsize=(6 * n_sp, 5), squeeze=False,
                        constrained_layout=True)
axs = axs[0]

panel = 0
for sp in SPECIES:
    series = spectra_by_species[sp]
    if len(series) == 0:
        continue
    ax = axs[panel]; panel += 1

    for (t_code, bins, counts) in series:
        # bin centres: if bins are EDGES (len = counts+1), take midpoints.
        if bins.size == counts.size + 1:
            x = 0.5 * (bins[1:] + bins[:-1])
            widths = np.diff(bins)
        else:
            x = bins
            # approximate widths from spacing for optional normalisation
            widths = np.gradient(bins)

        y = counts.astype(float)
        if NORMALISE_BY_BIN_WIDTH:
            with np.errstate(divide="ignore", invalid="ignore"):
                y = np.where(widths > 0, y / widths, 0.0)

        colour = cmap(norm(t_code)) if np.isfinite(t_code) else "gray"
        # mask non-positive for log-log
        good = (x > 0) & (y > 0)
        ax.plot(x[good], y[good], color=colour, lw=1.2, alpha=0.85)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\gamma - 1$", fontsize = 14)
    ax.set_ylabel(r"$dN/d\gamma$", fontsize = 14)
    ax.set_title(f"Species {sp} spectrum", fontsize = 14)

    # user-defined x-axis limits (only the side(s) provided)
    if (XMIN is not None) or (XMAX is not None):
        ax.set_xlim(left=XMIN, right=XMAX)

    # user-defined y-axis limits (only the side(s) provided)
    if (YMIN is not None) or (YMAX is not None):
        ax.set_ylim(bottom=YMIN, top=YMAX)
    
# shared colourbar for time
sm = cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])
cbar = fig.colorbar(sm, ax=list(axs[:panel]), fraction=0.046, pad=0.02)
cbar.set_label(r"$t\,\omega_{pe}$", fontsize = 14)

outfile = os.path.join(OUTDIR, "spectra.png")
fig.savefig(outfile, dpi=150, bbox_inches="tight")
plt.close(fig)

print(f"\nSaved {outfile}", flush=True)