"""
Created on Fri Sep 26 2026

@author: Pranab JD, Claude AI

Usage
-----
    fields="/scratch/.../RMR/fields"
    out="/scratch/.../RMR/plots"
    srun python3 -u ../postprocessing/Spectra_SF.py "$fields" "$out"     # compute + plot
    python3 ../postprocessing/Spectra_SF.py "$fields" "$out"             # replot (REPLOT_FROM_CACHE=True)
"""

import os
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import glob
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from adios2 import Stream

try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    HAVE_MPI = True
except Exception:                                   #! fall back to serial
    comm = None; rank = 0; size = 1; HAVE_MPI = False

#! ============================================================
#! USER SETTINGS
#! ============================================================
TIME_KEY      = "Time"          #! Entity time variable/attribute
B_COMPS       = ["fB1", "fB2", "fB3"]   #! magnetic vector components
J_COMPS       = ["fJ1", "fJ2", "fJ3"]   #! current vector components

SF_ORDER      = 4               #! structure-function order (even; reference fast path)
SF_N_LAGS     = 40              #! number of geometrically-spaced lags
SF_MIN_LAG    = 1               #! smallest lag (cells)  -- raise toward the filter/gyro scale if noisy
SF_MAX_FRAC   = 0.5             #! largest lag as a fraction of Nx (x is periodic)
CS_SLAB_FRAC  = 0.5             #! TOTAL y-fraction, centred on the sheet (Ly/2). Set 1.0 for whole box.

FILE_STRIDE   = 5               #! process every Nth snapshot (1 = all)   [COMPUTE stage]
CMAP          = "jet"           #! time colormap
ONE_SIDED     = True            #! double interior modes for a one-sided PSD

SPEC_SLOPE_GUIDE = -2.5         #! faint reference slope on the spectra (None to disable)

REPLOT_FROM_CACHE = True        #! True -> skip compute, redraw from CACHE_NAME (run serially)
CACHE_NAME        = "spectra_sf_cache.npz"

#! ---- which cached snapshots to DRAW (plot-only; does not affect compute/cache) ----
PLOT_STRIDE = 5                #! e.g. 4 -> draw every 4th cached snapshot (None or 1 = all)
PLOT_TIMES  = None              #! e.g. [0.5, 1.0, 2.0] -> nearest cached snapshot to each t c/Lx
                                #!      (overrides PLOT_STRIDE when set)
COLOR_ABSOLUTE_TIME = True      #! True -> colorbar spans the FULL cached time range, so a colour
                                #!         means the same t whether or not neighbours are drawn.
                                #! False -> colorbar spans only the drawn subset.

PLOT_TRANGE = (0.5, 3)         #! draw only snapshots with t c/Lx in this window; None = open end
                                #!   e.g. (None, 2.0) -> up to t=2 ; (1.0, 3.0) -> a window ; (None,None) -> all
COLORBAR_FILL_WINDOW = True     #! only if COLOR_ABSOLUTE_TIME: True -> colormap fills the DRAWN window
                                #!   (max contrast within it); False -> colours stay keyed to the full cached range

SPEC_XLIM = (7e-3, 3e1)     #! k-range for BOTH spectra panels, e.g. (2e-2, 3.0)
EB_YLIM   = (1e-8, 1e0)      #! E_B(k) y-range
EJ_YLIM   = (1e-7, 1e-1)     #! E_J(k) y-range
SF_XLIM   = (None, None)     #! lag-range for BOTH SF panels, e.g. (0.1, 100)
SFB_YLIM  = (None, None)     #! SF4 dB y-range
SFJ_YLIM  = (None, None)     #! SF4 J  y-range

#! ---- vertical reference lines at characteristic scales (read from Simulation_params.txt, in d0) ----
#! set a length to None to skip that line. On SPECTRA the line sits at the WAVENUMBER of that
#! length; on the SF at the LAG equal to that length.
SCALE_LINES = {
    "d_e":       (1.00756,  "blue"),        #! skin depth   (length in d0, colour)
    "rho":       (0.507593, "tab:green"),   #! Larmor radius
    "lambda_De": (0.100756, "tab:red"),     #! Debye length
}
SCALE_LINE_STYLE = "--"
K_FROM_LENGTH_2PI = True    #! spectra marker: True -> k = 2*pi/length (wavelength) ; False -> k = 1/length (k*l=1)

#! ============================================================
#! Args
#! ============================================================
parser = argparse.ArgumentParser()
parser.add_argument("base",   type=str, help="Directory with fields.*.bp")
parser.add_argument("outdir", type=str, nargs="?", default=None, help="Output dir (default: base)")
args   = parser.parse_args()
base   = args.base
outdir = args.outdir if args.outdir is not None else base
if rank == 0:
    os.makedirs(outdir, exist_ok=True)
if HAVE_MPI:
    comm.Barrier()

cache_path = os.path.join(outdir, CACHE_NAME)

#! ============================================================
#! Vertical scale-marker lines
#! ============================================================
SCALE_LABELS = {"d_e": r"$d_e$", "rho": r"$\rho$", "lambda_De": r"$\lambda_{De}$"}

def add_scale_lines(ax, kind):
    """Draw dashed vertical lines at the characteristic scales.
       kind='k'   (spectra, x=k):   position = 2*pi/length (or 1/length if K_FROM_LENGTH_2PI=False)
       kind='lag' (SF, x=length):   position = length."""
    for name, (length, color) in SCALE_LINES.items():
        if length is None or length <= 0:
            continue
        if kind == "k":
            pos = (2.0 * np.pi / length) if K_FROM_LENGTH_2PI else (1.0 / length)
        else:
            pos = length
        ax.axvline(pos, color=color, ls=SCALE_LINE_STYLE, lw=1.2, alpha=0.85,
                   label=SCALE_LABELS.get(name, name))

#! ============================================================
#! Plotting (shared by the compute path and the replot path)
#! ============================================================
def plot_all(times, k, lag_d, E_B_all, E_J_all, sf_B_all, sf_J_all):
    times   = np.asarray(times)
    E_B_all = np.asarray(E_B_all); E_J_all = np.asarray(E_J_all)
    sf_B_all = np.asarray(sf_B_all); sf_J_all = np.asarray(sf_J_all)

    #! full time range BEFORE subsetting (for absolute-time colours)
    t_full_min, t_full_max = float(times.min()), float(times.max())

    #! ---- choose which snapshots to draw ----
    if PLOT_TIMES is not None:
        sel = np.unique([int(np.argmin(np.abs(times - t))) for t in PLOT_TIMES])
    elif PLOT_STRIDE is not None and PLOT_STRIDE > 1:
        sel = np.arange(0, len(times), PLOT_STRIDE)
    else:
        sel = np.arange(len(times))

    #! restrict to a time window (composes with the stride/times selection above)
    lo = -np.inf if PLOT_TRANGE[0] is None else PLOT_TRANGE[0]
    hi =  np.inf if PLOT_TRANGE[1] is None else PLOT_TRANGE[1]
    sel = sel[(times[sel] >= lo) & (times[sel] <= hi)]
    if sel.size == 0:
        raise SystemExit(f"No snapshots in PLOT_TRANGE={PLOT_TRANGE} "
                         f"(cached range {times.min():.3g}..{times.max():.3g})")

    times    = times[sel]
    E_B_all  = E_B_all[sel];  E_J_all  = E_J_all[sel]
    sf_B_all = sf_B_all[sel]; sf_J_all = sf_J_all[sel]
    print(f"Drawing {len(times)} of the cached snapshots: "
          f"t c/Lx = {np.round(times, 3).tolist()}", flush=True)

    if COLOR_ABSOLUTE_TIME and not COLORBAR_FILL_WINDOW:
        norm = Normalize(vmin=t_full_min, vmax=t_full_max)     #! colours keyed to full cached range
    else:
        norm = Normalize(vmin=float(times.min()), vmax=float(times.max()))  #! fill the drawn window
    cmap = matplotlib.colormaps[CMAP]
    kpos = k > 0
    ord_s = str(SF_ORDER)

    #! ---- FIGURE 1: power spectra ----
    fig1, axs1 = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for i in range(len(times)):
        col = cmap(norm(times[i]))
        axs1[0].plot(k[kpos], E_B_all[i][kpos], color=col, lw=1.1, alpha=0.85)
        axs1[1].plot(k[kpos], E_J_all[i][kpos], color=col, lw=1.1, alpha=0.85)

    spec_labels = [(r"Magnetic fluctuation $\delta B$", r"$E_B(k)$"),
                   (r"Current fluctuation $J$",         r"$E_J(k)$")]

    refs = [E_B_all[-1], E_J_all[-1]]                 #! anchor guide to latest DRAWN curve, PER panel
    spec_ylims = [EB_YLIM, EJ_YLIM]

    for j, ax in enumerate(axs1):
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel(r"$k\ d_0$", fontsize=13)
        ax.set_ylabel(spec_labels[j][1], fontsize=13)
        ax.set_title(spec_labels[j][0], fontsize=14)
        ax.set_xlim(SPEC_XLIM[0], SPEC_XLIM[1])
        ax.set_ylim(spec_ylims[j][0], spec_ylims[j][1])
        if SPEC_SLOPE_GUIDE is not None:
            kk = k[kpos]; rr = refs[j][kpos]
            a = max(1, len(kk) // 20)                 #! anchor ~5% into the k-range
            if a < len(kk) and np.isfinite(rr[a]) and rr[a] > 0:
                guide = kk ** SPEC_SLOPE_GUIDE * (rr[a] / kk[a] ** SPEC_SLOPE_GUIDE)
                ax.plot(kk, guide, color="k", ls="--", lw=1.0, label=fr"$k^{{{SPEC_SLOPE_GUIDE:.2f}}}$")
        add_scale_lines(ax, "k")                      #! vertical markers at k of each scale
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=9)
    sm = cm.ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
    cbar = fig1.colorbar(sm, ax=axs1, fraction=0.046, pad=0.02)
    cbar.set_label(r"$t\,c/L_x$", fontsize=13)
    out1 = os.path.join(outdir, "Spectra_dB_J.png")
    fig1.savefig(out1, dpi=150, bbox_inches="tight"); plt.close(fig1)

    #! ---- FIGURE 2: 4th-order structure functions ----
    fig2, axs2 = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for i in range(len(times)):
        col = cmap(norm(times[i]))
        axs2[0].plot(lag_d, sf_B_all[i], color=col, lw=1.1, alpha=0.85)
        axs2[1].plot(lag_d, sf_J_all[i], color=col, lw=1.1, alpha=0.85)

    sf_ylab = r"$S_{" + ord_s + r"}(\ell)=\langle|\Delta \mathrm{field}|^{" + ord_s + r"}\rangle$"
    sf_ylims = [SFB_YLIM, SFJ_YLIM]

    for j, (ax, ttl) in enumerate(((axs2[0], r"$\delta B$"), (axs2[1], r"$J$"))):
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel(r"lag $\ell\ [d_0]$", fontsize=13)
        ax.set_ylabel(sf_ylab, fontsize=13)
        ax.set_title(ttl + fr"  (order {SF_ORDER} SF, along $x$)", fontsize=14)
        ax.set_xlim(SF_XLIM[0], SF_XLIM[1])
        ax.set_ylim(sf_ylims[j][0], sf_ylims[j][1])
        add_scale_lines(ax, "lag")                    #! vertical markers at lag = each scale
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=9)

    sm2 = cm.ScalarMappable(norm=norm, cmap=cmap); sm2.set_array([])
    cbar2 = fig2.colorbar(sm2, ax=axs2, fraction=0.046, pad=0.02)
    cbar2.set_label(r"$t\,c/L_x$", fontsize=13)
    out2 = os.path.join(outdir, "sf4_dB_J.png")
    fig2.savefig(out2, dpi=150, bbox_inches="tight"); plt.close(fig2)

    return out1, out2

#! ============================================================
#! REPLOT-ONLY PATH: skip compute, redraw from cache
#! ============================================================
if REPLOT_FROM_CACHE:
    if rank != 0:
        raise SystemExit(0)                          #! only rank 0 replots
    if not os.path.exists(cache_path):
        raise SystemExit(f"REPLOT_FROM_CACHE=True but no cache at {cache_path}")
    d = np.load(cache_path, allow_pickle=True)
    o1, o2 = plot_all(d["times"], d["k"], d["lag_d"],
                      d["E_B"], d["E_J"], d["sf_B"], d["sf_J"])
    print(f"Replotted from cache -> {o1}\n                        {o2}", flush=True)
    raise SystemExit(0)

#! ============================================================
#! Helpers (compute)
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

def build_sf_lags(n):
    hi = max(SF_MIN_LAG + 1, int(np.floor(SF_MAX_FRAC * n)))
    lags = np.unique(np.round(np.geomspace(SF_MIN_LAG, hi, num=SF_N_LAGS)).astype(int))
    return lags[(lags >= SF_MIN_LAG) & (lags <= hi)]

def x_spectrum(dfields, Nx, dx, half_factor):
    """1D power spectrum along periodic x, averaged over slab rows."""
    nk = Nx // 2 + 1
    P  = np.zeros(nk, dtype=np.float64)
    for g in dfields:
        F = np.fft.rfft(g, axis=1) / Nx
        p = (np.abs(F) ** 2)
        if ONE_SIDED:
            if Nx % 2 == 0:
                p[:, 1:-1] *= 2.0
            else:
                p[:, 1:] *= 2.0
        P += half_factor * p.mean(axis=0)
    k = 2.0 * np.pi * np.fft.rfftfreq(Nx, d=dx)
    return k, P

def x_sf_even(dfields, lags, order):
    """SF_order = < |df|^order > along periodic x (roll), averaged over x and rows."""
    half = order // 2
    out  = np.full(lags.shape, np.nan, dtype=np.float64)
    for li, l in enumerate(lags):
        d2 = None
        for g in dfields:
            d = (np.roll(g, -int(l), axis=1) - g).astype(np.float64)
            d2 = d * d if d2 is None else d2 + d * d
        out[li] = np.mean(d2 ** half)
    return out

def fluctuation(field2d):
    """df = f - <f>_x(y): subtract the x-mean per y-row."""
    return field2d - field2d.mean(axis=1, keepdims=True)

def process_file(fname):
    with Stream(fname, "r") as s:
        next(s.steps())
        x = np.asarray(s.read("X1")).ravel(); Nx = x.size
        y = np.asarray(s.read("X2")).ravel(); Ny = y.size
        try:
            z = np.asarray(s.read("X3"))
            if z is not None and np.asarray(z).size > 1:
                raise SystemExit("This script is 2D-only; got a 3D dump.")
        except SystemExit:
            raise
        except Exception:
            pass
        t_code = read_time(s)

        Lx = float(x.max() - x.min())
        Ly = float(y.max() - y.min())
        dx = Lx / Nx
        cs_y = 0.5 * (y.min() + y.max())

        half = 0.5 * CS_SLAB_FRAC * Ly
        jlo = int(np.searchsorted(y, cs_y - half, side="left"))
        jhi = int(np.searchsorted(y, cs_y + half, side="right"))
        jlo = max(0, jlo); jhi = min(Ny, max(jhi, jlo + 1))
        ny_slab = jhi - jlo

        Braw = [np.asarray(s.read(c, start=[jlo, 0], count=[ny_slab, Nx])) for c in B_COMPS]
        Jraw = [np.asarray(s.read(c, start=[jlo, 0], count=[ny_slab, Nx])) for c in J_COMPS]

    dB = [fluctuation(b) for b in Braw]
    dJ = [fluctuation(j) for j in Jraw]

    k, E_B = x_spectrum(dB, Nx, dx, half_factor=0.5)
    _, E_J = x_spectrum(dJ, Nx, dx, half_factor=1.0)

    lags   = build_sf_lags(Nx)
    sf_B   = x_sf_even(dB, lags, SF_ORDER)
    sf_J   = x_sf_even(dJ, lags, SF_ORDER)
    lag_d  = lags * dx

    t_lc = t_code / Lx if (np.isfinite(t_code) and Lx > 0) else float("nan")
    return dict(t_lc=t_lc, k=k, E_B=E_B, E_J=E_J, lag_d=lag_d, sf_B=sf_B, sf_J=sf_J)

#! ============================================================
#! Gather files, distribute across ranks
#! ============================================================
files = sorted(glob.glob(f"{base}/fields.*.bp"), key=step_from_fname)[::FILE_STRIDE]
if not files:
    raise SystemExit(f"No fields.*.bp in {base}")
if rank == 0:
    print(f"Found {len(files)} snapshots (stride {FILE_STRIDE}); slab = {CS_SLAB_FRAC:.2f} Ly around sheet", flush=True)
    if HAVE_MPI and size > len(files):
        print(f"NOTE: {size} tasks for {len(files)} files -> {size-len(files)} idle; use -n {len(files)}", flush=True)

my_files = files[rank::size]
my_results = []
for f in my_files:
    try:
        my_results.append(process_file(f))
        print(f"done {os.path.basename(f)}", flush=True)
    except SystemExit:
        raise
    except Exception as exc:
        print(f"SKIP {os.path.basename(f)}: {type(exc).__name__}: {exc}", flush=True)

if HAVE_MPI:
    gathered = comm.gather(my_results, root=0)
else:
    gathered = [my_results]

if rank != 0:
    raise SystemExit(0)

results = [r for chunk in gathered for r in chunk]
results = [r for r in results if np.isfinite(r["t_lc"])]
if not results:
    raise SystemExit("No usable snapshots (times missing?).")
results.sort(key=lambda r: r["t_lc"])

k        = results[0]["k"]
lag_d    = results[0]["lag_d"]
times    = np.array([r["t_lc"] for r in results])
E_B_all  = np.array([r["E_B"]  for r in results])
E_J_all  = np.array([r["E_J"]  for r in results])
sf_B_all = np.array([r["sf_B"] for r in results])
sf_J_all = np.array([r["sf_J"] for r in results])

#! ---- SAVE CACHE FIRST: compute is now safe even if plotting fails ----
np.savez_compressed(cache_path,
                    times=times, k=k, lag_d=lag_d,
                    E_B=E_B_all, E_J=E_J_all, sf_B=sf_B_all, sf_J=sf_J_all,
                    sf_order=SF_ORDER, cs_slab_frac=CS_SLAB_FRAC)
print(f"Wrote cache: {cache_path}", flush=True)

#! ---- plot (wrapped: a rendering error leaves the cache intact) ----
try:
    out1, out2 = plot_all(times, k, lag_d, E_B_all, E_J_all, sf_B_all, sf_J_all)
    print(f"Saved {out1}\nSaved {out2}", flush=True)
except Exception as exc:
    print(f"PLOTTING FAILED ({type(exc).__name__}: {exc}); cache is safe. "
          f"Fix the plot and rerun with REPLOT_FROM_CACHE=True.", flush=True)
    raise

print(f"  snapshots: {len(results)}   t c/Lx in [{times.min():.3g}, {times.max():.3g}]", flush=True)
print(f"  spectrum: periodic x, {CS_SLAB_FRAC:.2f}Ly slab; df=f-<f>_x(y)", flush=True)
print(f"  SF: order {SF_ORDER}, vector increment magnitude along x", flush=True)