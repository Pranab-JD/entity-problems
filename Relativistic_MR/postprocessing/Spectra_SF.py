"""
Created on Fri Sep 26 2026

@author: Pranab JD, Claude AI

Usage
-----
    fields="/scratch/.../RMR/fields"
    out="/scratch/.../RMR/plots"
    srun python3 -u ../postprocessing/Spectra_SF.py "$fields" "$out"     # compute + plot
    python3 ../postprocessing/Spectra_SF.py "$fields" "$out"             # replot (REPLOT_FROM_CACHE=True)

Structure functions follow Hu et al. (2026), arXiv:2512.12516:
    SF_2(dr) = < |f(x + dr) - f(x)|^2 >          (vector increment, all 3 components)
and it is SQRT(SF_2) that is plotted, so the Kolmogorov reference slope is 1/3
(velocity-like) and the magnetic reference is ~2/3
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

#! ---- structure functions (Hu+2026 convention) ----
SF_ORDER      = 2               #! DESCRIPTIVE ONLY -- sf2_x/sf2_y hardcode the 2nd
                                #! order. Changing this does NOT change the maths;
                                #! it is written to the cache purely as a label.
SF_N_LAGS     = 40              #! number of geometrically-spaced lags (per direction)
SF_MIN_LAG    = 1               #! smallest lag (cells) -- raise toward the filter/gyro scale if noisy
SF_MAX_FRAC   = 0.5             #! largest x-lag as a fraction of Nx (x is periodic -> roll)
SF_Y_MAX_FRAC = 0.5             #! largest y-lag as a fraction of the SLAB height (y is NOT periodic)
CS_SLAB_FRAC  = 0.5             #! TOTAL y-fraction, centred on the sheet (Ly/2). Set 1.0 for whole box.

#! Hu+2026 restrict to a "reconnection region" defined by a particle-mixing
#! criterion (both inflow populations >= 1% of local density). That needs
#! origin-tagged particles, which these dumps do not carry, so CS_SLAB_FRAC is a
#! fixed-slab PROXY for it -- a rectangular box rather than a mixing surface.
#! Consequence: the slab still contains un-reconnected upstream plasma near its
#! y-edges, which the paper's mask would have excluded.

SF_DETREND_Y  = True            #! subtract <f>_x(y) before the y-increments.
                                #! ALONG X this changes nothing (a per-row constant
                                #! cancels in the increment), but ALONG Y it removes
                                #! the Harris equilibrium B_x(y) = B0 tanh(...), which
                                #! would otherwise dominate SF_y at every lag. The
                                #! paper achieves the same end via its region mask.
                                #! Set False to see the raw (equilibrium-dominated) y-SF.

SF_FIT_RANGE  = (2.0, 30.0)     #! lag window [d0] for the reported log-log slope.
                                #! Hu+2026 fit 2-30 d_e. CHECK this is inside your
                                #! resolved range: below the current-filter scale the
                                #! slope is numerical, not physical.

FILE_STRIDE   = 5               #! process every Nth snapshot (1 = all)   [COMPUTE stage]
                                #! The FIRST and LAST files are force-included below,
                                #! whatever this is set to.
CMAP          = "jet"           #! time colormap
ONE_SIDED     = True            #! double interior modes for a one-sided PSD

SPEC_SLOPE_GUIDE = -2.5         #! faint reference slope on the spectra (None to disable)

REPLOT_FROM_CACHE = False        #! True -> skip compute, redraw from CACHE_NAME (run serially)
CACHE_NAME        = "spectra_sf2_cache.npz"   #! NOTE: renamed -- the old sf4 cache has a
                                              #! different key set and will not load here.

#! ---- which cached snapshots to DRAW (plot-only; does not affect compute/cache) ----
PLOT_STRIDE = 2                #! e.g. 4 -> draw every 4th cached snapshot (None or 1 = all)
PLOT_TIMES  = None              #! e.g. [0.5, 1.0, 2.0] -> nearest cached snapshot to each t c/Lx
                                #!      (overrides PLOT_STRIDE when set)
COLOR_ABSOLUTE_TIME = True      #! True -> colorbar spans the FULL cached time range, so a colour
                                #!         means the same t whether or not neighbours are drawn.
                                #! False -> colorbar spans only the drawn subset.

PLOT_TRANGE = (0.5, 3)          #! t c/Lx window; None = open end. This sets BOTH the drawn
                                #! range AND which snapshot the quoted slope comes from: the
                                #! latest cached one at or below the upper bound (force-included
                                #! even if PLOT_STRIDE skipped it). Raise the upper bound to
                                #! report a later time.
COLORBAR_FILL_WINDOW = True     #! only if COLOR_ABSOLUTE_TIME: True -> colormap fills the DRAWN window
                                #!   (max contrast within it); False -> colours stay keyed to the full cached range

SPEC_XLIM = (7e-3, 3e1)     #! k-range for BOTH spectra panels, e.g. (2e-2, 3.0)
EB_YLIM   = (1e-8, 1e0)      #! E_B(k) y-range
EJ_YLIM   = (1e-7, 1e-1)     #! E_J(k) y-range
SF_XLIM   = (None, None)     #! lag-range for ALL SF panels, e.g. (0.1, 100)
SFB_YLIM  = (None, None)     #! sqrt(SF2) dB y-range
SFJ_YLIM  = (None, None)     #! sqrt(SF2) J  y-range

SF_PLOT_J = False               #! PLOT-ONLY: False -> draw the dB row alone (2 panels).
                                #! J is still computed and cached either way, so this can
                                #! be flipped and redrawn with REPLOT_FROM_CACHE=True.

SF_GUIDES = [(1.0 / 3.0, "--",  r"$1/3$"),      #! Kolmogorov (velocity-like) reference
             (2.0 / 3.0, "-.",  r"$2/3$")]      #! magnetic reference seen by Hu+2026

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

def fit_loglog_slope(lag, s, lo, hi):
    """Least-squares log-log slope of s(lag) over lag in [lo, hi]. NaN if too few points."""
    lag = np.asarray(lag, dtype=np.float64)
    s   = np.asarray(s,   dtype=np.float64)
    m   = np.isfinite(s) & (s > 0) & (lag >= lo) & (lag <= hi)
    if m.sum() < 3:
        return float("nan")
    return float(np.polyfit(np.log10(lag[m]), np.log10(s[m]), 1)[0])

#! ============================================================
#! Plotting (shared by the compute path and the replot path)
#! ============================================================
def plot_all(times, k, lag_x, lag_y, E_B_all, E_J_all,
             sfx_B_all, sfx_J_all, sfy_B_all, sfy_J_all):
    times     = np.asarray(times)
    E_B_all   = np.asarray(E_B_all);   E_J_all   = np.asarray(E_J_all)
    sfx_B_all = np.asarray(sfx_B_all); sfx_J_all = np.asarray(sfx_J_all)
    sfy_B_all = np.asarray(sfy_B_all); sfy_J_all = np.asarray(sfy_J_all)

    #! full time range BEFORE subsetting (for absolute-time colours)
    t_full_min, t_full_max = float(times.min()), float(times.max())

    #! ---- keep the FIRST cached snapshot aside -------------------------
    #! Its slope is printed, but it is NOT drawn: it is the pre-turbulent
    #! baseline, and on a log axis it sits decades below the rest and would
    #! collapse the y-range of every panel. (results were sorted by time
    #! before caching, so index 0 is the earliest.)
    t_first     = float(times[0])
    sfx_B_first = sfx_B_all[0].copy(); sfx_J_first = sfx_J_all[0].copy()
    sfy_B_first = sfy_B_all[0].copy(); sfy_J_first = sfy_J_all[0].copy()

    #! ---- choose which snapshots to draw ----
    if PLOT_TIMES is not None:
        sel = np.unique([int(np.argmin(np.abs(times - t))) for t in PLOT_TIMES])
    elif PLOT_STRIDE is not None and PLOT_STRIDE > 1:
        sel = np.arange(0, len(times), PLOT_STRIDE)
    else:
        sel = np.arange(len(times))

    #! restrict to the PLOT_TRANGE window (composes with the stride/times selection)
    lo = -np.inf if PLOT_TRANGE[0] is None else PLOT_TRANGE[0]
    hi =  np.inf if PLOT_TRANGE[1] is None else PLOT_TRANGE[1]
    sel = sel[(times[sel] >= lo) & (times[sel] <= hi)]

    #! The REPORTED snapshot: latest cached one at or BELOW the upper bound.
    #! Force-included so a stride cannot skip it -- it is the curve the quoted
    #! slope and the guide lines refer to, and after np.unique it is last in sel.
    in_window = np.nonzero(times <= hi)[0]
    if in_window.size == 0:
        raise SystemExit(f"No cached snapshot at or below PLOT_TRANGE[1]={PLOT_TRANGE[1]} "
                         f"(cached range {t_full_min:.3g}..{t_full_max:.3g})")
    i_report = int(in_window[-1])
    sel = np.unique(np.concatenate((sel, [i_report])).astype(int))

    #! drop the first cached snapshot from the DRAWN set (its slope is still reported)
    sel = sel[sel != 0] if sel.size > 1 else sel

    times     = times[sel]
    E_B_all   = E_B_all[sel];   E_J_all   = E_J_all[sel]
    sfx_B_all = sfx_B_all[sel]; sfx_J_all = sfx_J_all[sel]
    sfy_B_all = sfy_B_all[sel]; sfy_J_all = sfy_J_all[sel]
    print(f"Drawing {len(times)} snapshots (t={t_first:.3g} excluded from the plot): "
          f"t c/Lx = {np.round(times, 3).tolist()}", flush=True)

    if COLOR_ABSOLUTE_TIME and not COLORBAR_FILL_WINDOW:
        norm = Normalize(vmin=t_full_min, vmax=t_full_max)     #! colours keyed to full cached range
    else:
        norm = Normalize(vmin=float(times.min()), vmax=float(times.max()))  #! fill the drawn window
    cmap = matplotlib.colormaps[CMAP]
    kpos = k > 0

    #! ---- FIGURE 1: power spectra  (UNCHANGED) ----
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

    #! ---- FIGURE 2: sqrt of 2nd-order structure functions, x and y ----
    #! Layout: rows = field (dB, J), columns = separation direction (x, y).
    #! Plotted quantity is sqrt(SF_2), matching Hu+2026, so the guide slopes
    #! 1/3 and 2/3 are directly comparable to their Figs. 2-4.
    n_rows = 2 if SF_PLOT_J else 1
    fig2, axs2 = plt.subplots(n_rows, 2, figsize=(13, 5 * n_rows),
                              constrained_layout=True, squeeze=False)

    #! tuple: (axis, lags, drawn SF array, FIRST-snapshot SF, field label, direction, ylim)
    panels = [
        (axs2[0, 0], lag_x, sfx_B_all, sfx_B_first, r"$\delta B$", r"$\Delta x$  (outflow)", SFB_YLIM),
        (axs2[0, 1], lag_y, sfy_B_all, sfy_B_first, r"$\delta B$", r"$\Delta y$  (inflow)",  SFB_YLIM)]
    if SF_PLOT_J:
        panels += [
            (axs2[1, 0], lag_x, sfx_J_all, sfx_J_first, r"$J$", r"$\Delta x$  (outflow)", SFJ_YLIM),
            (axs2[1, 1], lag_y, sfy_J_all, sfy_J_first, r"$J$", r"$\Delta y$  (inflow)",  SFJ_YLIM)]

    fit_lo, fit_hi = SF_FIT_RANGE
    print(f"\nsqrt(SF_2) log-log slopes, fitted over lag in [{fit_lo:g}, {fit_hi:g}] d0.", flush=True)
    print(f"   first cached t={t_first:.3g} (printed only, not drawn) | "
          f"reported t={times[-1]:.3g} (<= PLOT_TRANGE[1]={PLOT_TRANGE[1]})", flush=True)

    for ax, lag, sf_all, sf_first, fld, dirn, ylim in panels:
        root = np.sqrt(np.asarray(sf_all))            #! sqrt(SF_2): the plotted quantity

        for i in range(len(times)):
            ax.plot(lag, root[i], color=cmap(norm(times[i])), lw=1.1, alpha=0.85)

        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel(r"lag $\ell\ [d_0]$", fontsize=13)
        ax.set_ylabel(r"$\sqrt{\mathrm{SF}_2(\ell)}$", fontsize=13)
        ax.set_title(f"{fld}  along {dirn}", fontsize=14)
        ax.set_xlim(SF_XLIM[0], SF_XLIM[1])
        ax.set_ylim(ylim[0], ylim[1])

        #! reference power laws, anchored to the REPORTED curve inside the fit window
        ref = root[-1]
        m   = np.isfinite(ref) & (ref > 0) & (lag >= fit_lo) & (lag <= fit_hi)
        if m.any():
            a = int(np.argmax(m))                     #! first in-window point
            for sl, ls, lab in SF_GUIDES:
                g = lag ** sl * (ref[a] / lag[a] ** sl)
                ax.plot(lag, g, color="k", ls=ls, lw=1.0, label=lab)

        #! slopes: the first cached snapshot goes to stdout only; the reported one
        #! (last drawn, i.e. latest at or below PLOT_TRANGE[1]) is also annotated.
        s_first  = fit_loglog_slope(lag, np.sqrt(np.asarray(sf_first)), fit_lo, fit_hi)
        s_report = fit_loglog_slope(lag, ref, fit_lo, fit_hi)
        dir_tag  = "x" if lag is lag_x else "y"
        print(f"   {fld:>10s}  along {dir_tag:<3s} : "
              f"t={t_first:.3g} -> {s_first:.3f}    "
              f"t={times[-1]:.3g} -> {s_report:.3f}", flush=True)

        ax.text(0.03, 0.95, fr"slope $= {s_report:.2f}$  ($t={times[-1]:.2f}$)",
                transform=ax.transAxes, fontsize=13, va="top", ha="left")

        add_scale_lines(ax, "lag")                    #! vertical markers at lag = each scale
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=9, loc="lower right")

    sm2 = cm.ScalarMappable(norm=norm, cmap=cmap); sm2.set_array([])
    cbar2 = fig2.colorbar(sm2, ax=axs2, fraction=0.046, pad=0.02)
    cbar2.set_label(r"$t\,c/L_x$", fontsize=13)
    out2 = os.path.join(outdir, "sf2_dB_J.png")
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
    o1, o2 = plot_all(d["times"], d["k"], d["lag_x"], d["lag_y"],
                      d["E_B"], d["E_J"],
                      d["sfx_B"], d["sfx_J"], d["sfy_B"], d["sfy_J"])
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

def build_sf_lags(n, max_frac):
    """Geometrically-spaced integer lags in [SF_MIN_LAG, max_frac*n]."""
    hi = max(SF_MIN_LAG + 1, int(np.floor(max_frac * n)))
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

def sf2_x(fields, lags):
    """SF_2(dx) = < |f(x+dx) - f(x)|^2 > along PERIODIC x.

    Vector increment: the three components are summed BEFORE averaging, i.e.
    |Df|^2 = Df1^2 + Df2^2 + Df3^2, matching Hu+2026 Eq. 1. x is periodic in
    these runs, so np.roll is a legitimate wrap; if you ever run open-x, this
    must become a truncated increment like sf2_y below.
    """
    out = np.full(lags.shape, np.nan, dtype=np.float64)
    for li, l in enumerate(lags):
        acc = None
        for g in fields:
            d = (np.roll(g, -int(l), axis=1) - g).astype(np.float64)
            acc = d * d if acc is None else acc + d * d
        out[li] = np.mean(acc)                        #! average over x and over rows
    return out

def sf2_y(fields, lags):
    """SF_2(dy) = < |f(y+dy) - f(y)|^2 > along NON-PERIODIC y.

    y is reflecting and carries the Harris equilibrium, so NO wrap: the
    increment is truncated to pairs that both lie inside the slab. That means
    the number of contributing pairs shrinks as the lag grows -- large-lag
    points are noisier, and lags beyond SF_Y_MAX_FRAC of the slab are not
    computed at all.
    """
    ny  = fields[0].shape[0]
    out = np.full(lags.shape, np.nan, dtype=np.float64)
    for li, l in enumerate(lags):
        l = int(l)
        if l >= ny:                                   #! lag exceeds the slab: leave as NaN
            continue
        acc = None
        for g in fields:
            d = (g[l:, :] - g[:-l, :]).astype(np.float64)
            acc = d * d if acc is None else acc + d * d
        out[li] = np.mean(acc)
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
        dy = Ly / Ny                                  #! cell size along y (uniform Minkowski)
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

    #! ---- power spectra (unchanged) ----
    k, E_B = x_spectrum(dB, Nx, dx, half_factor=0.5)
    _, E_J = x_spectrum(dJ, Nx, dx, half_factor=1.0)

    #! ---- structure functions ----
    #! Along x the detrended and raw fields give IDENTICAL results (a per-row
    #! constant cancels in the increment), so dB/dJ are used for both cases.
    #! Along y the choice matters -- see SF_DETREND_Y.
    By = dB if SF_DETREND_Y else Braw
    Jy = dJ if SF_DETREND_Y else Jraw

    lags_x = build_sf_lags(Nx,      SF_MAX_FRAC)
    lags_y = build_sf_lags(ny_slab, SF_Y_MAX_FRAC)

    sfx_B = sf2_x(dB, lags_x);  sfx_J = sf2_x(dJ, lags_x)
    sfy_B = sf2_y(By, lags_y);  sfy_J = sf2_y(Jy, lags_y)

    lag_x = lags_x * dx
    lag_y = lags_y * dy                               #! y-lags use dy, NOT dx

    t_lc = t_code / Lx if (np.isfinite(t_code) and Lx > 0) else float("nan")
    return dict(t_lc=t_lc, k=k, E_B=E_B, E_J=E_J,
                lag_x=lag_x, lag_y=lag_y,
                sfx_B=sfx_B, sfx_J=sfx_J, sfy_B=sfy_B, sfy_J=sfy_J)

#! ============================================================
#! Gather files, distribute across ranks
#! ============================================================
all_files = sorted(glob.glob(f"{base}/fields.*.bp"), key=step_from_fname)
if not all_files:
    raise SystemExit(f"No fields.*.bp in {base}")

#! HARDCODED: the stride starts at index 0 so the FIRST file is always in; the LAST
#! file is appended when the stride would otherwise skip it. Both endpoints are
#! therefore always computed and cached, whatever FILE_STRIDE is set to. Whether
#! the last one is DRAWN or REPORTED is then decided by PLOT_TRANGE.
files = all_files[::FILE_STRIDE]
if files[-1] != all_files[-1]:
    files.append(all_files[-1])

if rank == 0:
    print(f"Found {len(all_files)} snapshots; processing {len(files)} "
          f"(stride {FILE_STRIDE}, first and last forced); "
          f"slab = {CS_SLAB_FRAC:.2f} Ly around sheet", flush=True)
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
results.sort(key=lambda r: r["t_lc"])       #! index 0 = earliest; the plot logic relies on this

k         = results[0]["k"]
lag_x     = results[0]["lag_x"]
lag_y     = results[0]["lag_y"]
times     = np.array([r["t_lc"]  for r in results])
E_B_all   = np.array([r["E_B"]   for r in results])
E_J_all   = np.array([r["E_J"]   for r in results])
sfx_B_all = np.array([r["sfx_B"] for r in results])
sfx_J_all = np.array([r["sfx_J"] for r in results])
sfy_B_all = np.array([r["sfy_B"] for r in results])
sfy_J_all = np.array([r["sfy_J"] for r in results])

#! ---- SAVE CACHE FIRST: compute is now safe even if plotting fails ----
np.savez_compressed(cache_path,
                    times=times, k=k, lag_x=lag_x, lag_y=lag_y,
                    E_B=E_B_all, E_J=E_J_all,
                    sfx_B=sfx_B_all, sfx_J=sfx_J_all,
                    sfy_B=sfy_B_all, sfy_J=sfy_J_all,
                    sf_order=SF_ORDER, cs_slab_frac=CS_SLAB_FRAC,
                    sf_detrend_y=SF_DETREND_Y)
print(f"Wrote cache: {cache_path}", flush=True)

#! ---- plot (wrapped: a rendering error leaves the cache intact) ----
try:
    out1, out2 = plot_all(times, k, lag_x, lag_y, E_B_all, E_J_all,
                          sfx_B_all, sfx_J_all, sfy_B_all, sfy_J_all)
    print(f"Saved {out1}\nSaved {out2}", flush=True)
except Exception as exc:
    print(f"PLOTTING FAILED ({type(exc).__name__}: {exc}); cache is safe. "
          f"Fix the plot and rerun with REPLOT_FROM_CACHE=True.", flush=True)
    raise

print(f"  snapshots: {len(results)}   t c/Lx in [{times.min():.3g}, {times.max():.3g}]", flush=True)
print(f"  spectrum: periodic x, {CS_SLAB_FRAC:.2f}Ly slab; df=f-<f>_x(y)", flush=True)
print(f"  SF: order 2 (sqrt plotted), vector increment, along x (periodic roll) "
      f"and y (truncated, detrend={SF_DETREND_Y})", flush=True)