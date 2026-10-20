"""
Created on Sat Jun 13 2026

@author: Pranab JD, Claude AI

Plot particle spectra dN/dln(gamma-1) from entity's spectra.*.bp files.
One panel per species, lines overlaid and coloured by t c / L_x.

Usage
-----
    spectra="/scratch/project_465003132/RMR_pair_3D/sigma_10_Z100/RMR/spectra"
    output="/scratch/project_465003132/RMR_pair_3D/sigma_10_Z100/RMR/plots"

    srun -N 1 -n 1 python3 -u ../postprocessing/Plot_spectra.py "$spectra" "$output" \
        --Lx 256 --R 64 --larmor0 0.5 --pair

    Required:
        --Lx          box length along X in code units (sets the t c/L_x normalisation)
        --larmor0     [scales] larmor0, for the gyroradius axis along the top

    Optional:
        --R           tube radius in code units; adds t c/R to the figure title
        --pair        species 2 is a POSITRON (light limits); default treats it as an ION
        --mass-ratio  species 2 / species 1 mass, for its gyroradius axis (default 1)
        --species     comma-separated species indices (default "1,2")
        --list_vars   print the BP variable names and exit
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
#! Axis limits; None -> auto on that side. The Y limits are shared by every species.
#! Y is a RATIO to the initial peak, not a count, so 1.0 is the t = 0 peak.
Y_MIN, Y_MAX           = 1e-4, 2.0
XLIGHT_MIN, XLIGHT_MAX = 1e-4, 1e3        #! electrons and positrons
XION_MIN, XION_MAX     = 5e-5, 1e0        #! ions

TIME_KEY   = "Time"
BIN_VARS   = ["sEbn", "ebins", "e_bins", "bins", "energy", "gamma"]        #! first match wins
COUNT_VARS = ["sN_{s}", "sN{s}", "spectrum_{s}", "f_{s}", "N{s}"]          #! {s} = species index

#! ============================================================
#! Args
#! ============================================================
parser = argparse.ArgumentParser()
parser.add_argument("base",         type=str)
parser.add_argument("outdir",       type=str)
parser.add_argument("--Lx",         type=float, required=True)
parser.add_argument("--R",          type=float, default=None, help="tube radius in code units; adds t c/R to the title")
parser.add_argument("--larmor0",    type=float, required=True)
parser.add_argument("--mass-ratio", dest="mass_ratio", type=float, default=1.0)
parser.add_argument("--species",    type=str, default="1,2")
parser.add_argument("--pair",       action="store_true")
parser.add_argument("--list_vars",  action="store_true")

args = parser.parse_args()

BASE, OUTDIR = args.base, args.outdir
SPECIES      = [int(s) for s in args.species.split(",") if s.strip() != ""]
IS_PAIR      = args.pair
LX, LARMOR0  = float(args.Lx), float(args.larmor0)
R_TUBE       = float(args.R) if args.R is not None else None
MR           = float(args.mass_ratio)

if LX <= 0.0:
    parser.error("--Lx must be positive")
if LARMOR0 <= 0.0:
    parser.error("--larmor0 must be positive")
if R_TUBE is not None and R_TUBE <= 0.0:
    parser.error("--R must be positive")

os.makedirs(OUTDIR, exist_ok=True)

files = sorted(glob.glob(f"{BASE}/spectra.*.bp"))
if len(files) == 0:
    raise SystemExit(f"No spectra.*.bp in {BASE}   (is [output.spectra] enable = true in the TOML?)")

#! ============================================================
#! Helpers
#! ============================================================
def read_time(stream):
    """Simulation time in code units: variable first, then attribute."""
    for getter in (lambda: stream.read(TIME_KEY), lambda: stream.read_attribute(TIME_KEY)):
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
        return []

def pick_var(names, candidates, s=None):
    """First candidate present in the file; {s} is filled with the species index when given."""
    for cand in candidates:
        name = cand.format(s=s) if s is not None else cand
        if name in names:
            return name
    return None

def species_info(sp):
    """
    Label, X limits and the gyroradius scaling for one species index. Species 1 is the electron;
    species 2 is a positron with --pair and an ion otherwise. r_L = larmor0 * (m/m_e) * u, so the heavy
    species needs the mass ratio in its top axis.
    """
    if sp == 1:
        return "Electron", XLIGHT_MIN, XLIGHT_MAX, 1.0
    if sp == 2:
        return ("Positron", XLIGHT_MIN, XLIGHT_MAX, 1.0) if IS_PAIR else ("Ions", XION_MIN, XION_MAX, MR)
    return f"Species {sp}", XLIGHT_MIN, XLIGHT_MAX, 1.0

def dN_dlng(bins, counts):
    """
    Bin centres, dN/dln(gamma-1), and the mask of plottable bins for one spectrum.
    Factored out so a curve and the reference it is normalised by are computed identically.
    """
    x = 0.5 * (bins[1:] + bins[:-1]) if bins.size == counts.size + 1 else bins
    with np.errstate(divide="ignore", invalid="ignore"):
        y = counts / np.gradient(np.log(x))      #! counts per bin -> dN/dln(gamma-1)
    good = (x > 0) & (counts > 0) & np.isfinite(y)
    return x, y, good

if args.list_vars:
    with Stream(files[0], "r") as s:
        next(s.steps())
        names = available_vars(s)
    print(f"\nVariables in {os.path.basename(files[0])}:")
    for n in names:
        print("   ", n)
    print("\nEdit BIN_VARS / COUNT_VARS at the top if these do not match.\n")
    raise SystemExit(0)

#! ============================================================
#! Read every file:  spectra[sp] = list of (t c/L_x, bins, counts)
#! ============================================================
print(f"\nFound {len(files)} spectra files   (L_x = {LX:g}, larmor0 = {LARMOR0:g})", flush=True)

spectra    = {sp: [] for sp in SPECIES}
times      = []                                   #! t c / L_x, used for the colour scale
times_code = []                                   #! raw t in code units, used for the title

for fname in files:
    with Stream(fname, "r") as s:
        next(s.steps())
        names  = available_vars(s)
        t_code = read_time(s)                        #! NaN propagates if Time is missing
        t_lc   = t_code / LX

        bin_var = pick_var(names, BIN_VARS)
        if bin_var is None:
            print(f"  [skip] {os.path.basename(fname)}: no bin array found", flush=True)
            continue
        bins = np.asarray(s.read(bin_var)).ravel()

        for sp in SPECIES:
            cvar = pick_var(names, COUNT_VARS, sp)
            if cvar is not None:
                spectra[sp].append((t_lc, bins, np.asarray(s.read(cvar)).ravel()))
    times.append(t_lc)
    times_code.append(t_code)

for sp in SPECIES:
    print(f"    species {sp}: {len(spectra[sp])} spectra", flush=True)

#! ============================================================
#! Plot: one panel per species, shared colourbar in t c / L_x
#! ============================================================
finite = [t for t in times if np.isfinite(t)]
norm   = Normalize(vmin=min(finite) if finite else 0.0, vmax=max(finite) if finite else 1.0)
cmap   = cm.jet

to_plot = [sp for sp in SPECIES if len(spectra[sp]) > 0]
if len(to_plot) == 0:
    raise SystemExit("No spectra read -- run with --list_vars and fix the variable names.")

fig, axs = plt.subplots(1, len(to_plot), figsize=(6 * len(to_plot), 5), squeeze=False, constrained_layout=True)
axs = axs[0]

for ax, sp in zip(axs, to_plot):
    label, xlo, xhi, mass = species_info(sp)

    #! Normalisation reference: peak of the EARLIEST spectrum for this species
    t0, bins0, counts0 = spectra[sp][0]
    _, y0, good0       = dN_dlng(bins0, counts0)
    ref_peak           = y0[good0].max() if np.any(good0) else 0.0
    if not np.isfinite(ref_peak) or ref_peak <= 0.0:
        print(f"  [warn] species {sp}: initial peak is {ref_peak!r}; plotting unnormalised", flush=True)
        ref_peak = 1.0
    print(f"    species {sp}: normalising by initial peak = {ref_peak:.6g}   (t c/L_x = {t0:.4g})", flush=True)

    for (t_lc, bins, counts) in spectra[sp]:
        x, y, good = dN_dlng(bins, counts)
        y = y / ref_peak                         #! <- normalised to the initial peak
        ax.plot(x[good], y[good], color=cmap(norm(t_lc)) if np.isfinite(t_lc) else "gray", lw=1.2, alpha=0.85)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\gamma - 1$", fontsize=14)
    ax.set_ylabel(r"$dN/d\ln(\gamma-1)$", fontsize=12)
    ax.set_title(f"{label} spectrum", fontsize=14)
    ax.set_xlim(left=xlo, right=xhi)
    ax.set_ylim(bottom=Y_MIN, top=Y_MAX)

    #! top axis: the relativistic gyroradius of a particle at that energy, r_L = larmor0 * (m/m_e) * u
    #! with u = sqrt(gamma^2 - 1). Valid at B = B_BG; inside the sheet the local field is smaller.
    rL   = lambda gm1: LARMOR0 * mass * np.sqrt((gm1 + 1.0) ** 2 - 1.0)
    gm1  = lambda r:   np.sqrt(1.0 + (r / (LARMOR0 * mass)) ** 2) - 1.0
    secax = ax.secondary_xaxis("top", functions=(rL, gm1))
    secax.set_xlabel(r"$r_L/d_0$", fontsize=14)

sm = cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])
fig.colorbar(sm, ax=list(axs), fraction=0.046, pad=0.02).set_label(r"$t\,c/L_x$", fontsize=14)

outfile = os.path.join(OUTDIR, "spectra.png")
fig.savefig(outfile, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nSaved {outfile}\n", flush=True)