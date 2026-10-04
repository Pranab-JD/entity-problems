"""
Created on Fri Oct 02 2026
@author: Pranab JD, Claude AI

Consistency diagnostics for the relativistic single-Harris-sheet Entity run.
Companion to Plot_B_J_rho.py -- same I/O conventions, same geometry assumptions.

One figure, five panels:
    (a) <J_z>_x against (curl B)_z at t = 0, the latter rescaled by the best-fit scale -- a SHAPE test
    (b) that scale over time; it must stay at AMP_COEFF, or the written J and B are not consistent
    (c) peak current, the same peak predicted from the written density, and the layer HWHM
    (d) E_z(y, t): the reconnection field, plus the start-up pulse and any wall reflection
    (e) the reconnection rate E_z/(B0 v_A) at the sheet centre, with v_A from sigma_tot = 2*sigma0/(1+mr)

What the report means
---------------------
Entity integrates dE/dt = curl B - J/AMP_COEFF with AMP_COEFF = skindepth0^2/larmor0, so the written J is
AMP_COEFF times curl B in equilibrium. Pass --larmor0, or every test below compares against a raw J.

The written J is smoothed by the current filter and the written density is not, so the two disagree at the
peak by a fixed fraction while conserving their integrals. The report calls that out as smoothing rather
than as an inconsistency; a genuine normalisation error shows up as a much larger, shape-preserving offset.

dE_z/dt is a finite difference ACROSS DUMPS. If the dumps are spaced by more than ~1 code time the
derivative is aliased and (g) cannot be read quantitatively; the report says so instead of giving a number.

Geometry (single Harris, reversal across Y): outflow = X, inflow = Y, guide = Z.
Array axis order: 2D (Ny, Nx) = (x2, x1); 3D (Nz, Ny, Nx) = (x3, x2, x1). All profiles are 1D in Y,
averaged over X and, in 3D, taken on the mid-Z plane only (a hyperslab read, never the whole cube).

Usage
-----
    srun -n 8 python3 Check_CS_transient.py "$input" "$output" --larmor0 0.3162 --cs-width 0.5 --cs-density 20
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")

from mpi4py import MPI
from adios2 import Stream

import argparse, os, glob
import matplotlib.pyplot as plt

#! ============================================================
#! USER SETTINGS
#! ============================================================

TIME_KEY = "Time"          #! adjust if Entity stores time under another key
F_BX = "fB1"               #! in-plane reversing field      B_x
F_BY = "fB2"               #! in-plane transverse field     B_y   (full 2D curl, div B)
F_JZ = "fJ3"               #! out-of-plane current          J_z
F_EZ = "fE3"               #! out-of-plane electric field   E_z   (needs "E" in output)
F_N  = "fN"                #! number density                      (needs "N" in output)

CURL_FRAC  = 0.10          #! fit only where |curl B| > CURL_FRAC * max
CORR_TOL   = 0.99          #! correlation above this = the two profiles are the same shape
SCALE_TOL  = 0.05          #! |s/AMP_COEFF - 1| below this = the law holds
SMOOTH_TOL = 0.35          #! peak deficits below this are attributed to the current filter
DT_TOL     = 1.0           #! dump spacing above this (code time) makes dE_z/dt unusable
DIVB_TOL   = 1.0e-2        #! |div B|/|curl B| above this is worth reporting

#! ============================================================
#! MPI setup
#! ============================================================

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

#! ============================================================
#! Args
#! ============================================================

parser = argparse.ArgumentParser()
parser.add_argument("base", type=str, help="Directory with fields.NNNNNNNNN.bp files")
parser.add_argument("outdir", type=str, help="Output directory for the PNG + CSV")
parser.add_argument("--larmor0", type=float, default=None, help="[scales] larmor0. Sets AMP_COEFF = skindepth0^2/larmor0. Omit for AMP_COEFF = 1.")
parser.add_argument("--skindepth0", type=float, default=1.0, help="[scales] skindepth0 (default 1.0). Only needed when it is not 1, e.g. an ion-electron run.")
parser.add_argument("--mass-ratio", dest="mass_ratio", type=float, default=1.0, help="species[1].mass / species[0].mass (default 1.0 = pair). Sets sigma_tot, hence v_A, hence the reconnection rate.")
parser.add_argument("--cs-width", dest="cs_width", type=float, default=None, help="setup.cs_width from the TOML, for the drift closure and the HWHM reference line.")
parser.add_argument("--cs-density", dest="cs_density", type=float, default=None, help="setup.cs_density (total) from the TOML, for the drift closure.")
parser.add_argument("--beta-d", dest="beta_d", type=float, default=None, help="Drift 3-velocity. Default: AMP_COEFF * B0 / (cs_density * cs_width) with B0 measured at t = 0.")
parser.add_argument("--n-factor", dest="n_factor", type=float, default=1.0, help="Multiply fN by this: 1.0 if it is the TOTAL density of both species, 2.0 if it is per species.")
parser.add_argument("--ywin", type=float, default=10.0, help="Half-width (code units) of the Y window used everywhere (default: 10)")
args = parser.parse_args()

base, outdir = args.base, args.outdir

if rank == 0:
    os.makedirs(outdir, exist_ok=True)
comm.Barrier()

files = sorted(glob.glob(f"{base}/fields.*.bp"))
files_local = files[rank::size]

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
    try:
        val = stream.read(TIME_KEY)
        if val is not None:
            return float(val)
    except Exception:
        pass
    try:
        return float(stream.read_attribute(TIME_KEY))
    except Exception:
        pass
    return float("nan")

def spacing(coord):
    """Median cell size of a cell-centred 1D coordinate array."""
    c = np.asarray(coord, dtype=float).ravel()
    return 1.0 if c.size < 2 else float(np.median(np.diff(c)))

def axis_span(coord):
    """Physical span of one axis, robust to output downsampling (which changes dc but not the span)."""
    c = np.asarray(coord, dtype=float).ravel()
    if c.size < 2:
        return float("nan")
    dc = spacing(c)
    return float((c[-1] + 0.5 * dc) - (c[0] - 0.5 * dc))

def read_plane(stream, name, is3d, k, Ny, Nx):
    """ONE (Ny, Nx) plane: a mid-Z hyperslab in 3D, the whole array in 2D. None if the variable is absent."""
    try:
        if is3d:
            a = np.asarray(stream.read(name, start=[int(k), 0, 0], count=[1, int(Ny), int(Nx)]))
            return a.reshape(int(Ny), int(Nx))
        return np.asarray(stream.read(name)).reshape(int(Ny), int(Nx))
    except Exception:
        return None

def half_width_half_max(prof_y, y):
    """HWHM of |prof| about its extremum, by linear interpolation on each flank. NaN if never crossed."""
    p = np.abs(np.asarray(prof_y, dtype=float))
    if p.size < 3:
        return float("nan")
    i0, half = int(np.argmax(p)), 0.5 * float(np.max(np.abs(prof_y)))
    if not np.isfinite(half) or half <= 0.0:
        return float("nan")
    def cross(idx_range):
        prev = i0
        for i in idx_range:
            if p[i] < half:
                f = (p[prev] - half) / max(p[prev] - p[i], 1.0e-300)
                return y[prev] + f * (y[i] - y[prev])
            prev = i
        return float("nan")
    y_lo, y_hi = cross(range(i0 - 1, -1, -1)), cross(range(i0 + 1, p.size))
    return float("nan") if not (np.isfinite(y_lo) and np.isfinite(y_hi)) else 0.5 * (y_hi - y_lo)

def curl_z(Bx, By, dx, dy):
    """(curl B)_z = dB_y/dx - dB_x/dy on the full plane, THEN averaged over x. Falls back to -dB_x/dy."""
    c = -np.gradient(np.asarray(Bx, dtype=float), dy, axis=0)
    if By is not None:
        c = c + np.gradient(np.asarray(By, dtype=float), dx, axis=1)
    return np.mean(c, axis=1)

def div_b(Bx, By, dx, dy):
    """In-plane |div B|_max. In 3D the dB_z/dz term is missing, so this is a partial check there."""
    if By is None:
        return float("nan")
    d = np.gradient(np.asarray(Bx, dtype=float), dx, axis=1) + np.gradient(np.asarray(By, dtype=float), dy, axis=0)
    return float(np.max(np.abs(d)))

def scale_and_corr(J, C):
    """Least-squares scale s = argmin |J - s*C|^2 (no intercept) and the Pearson correlation of the shapes."""
    J, C = np.asarray(J, dtype=float), np.asarray(C, dtype=float)
    den = float(np.sum(C * C))
    s   = float(np.sum(J * C) / den) if den > 0.0 else float("nan")
    if J.size < 3 or np.std(J) == 0.0 or np.std(C) == 0.0:
        return s, float("nan")
    return s, float(np.corrcoef(J, C)[0, 1])

#! ============================================================
#! Pass 1: each rank reduces its files to 1D-in-Y profiles
#! ============================================================

records = []

for fname in files_local:
    with Stream(fname, "r") as s:
        next(s.steps())
        x = np.asarray(s.read("X1")); Nx = x.size
        y = np.asarray(s.read("X2")); Ny = y.size
        try:
            z = np.asarray(s.read("X3")); Nz = z.size
        except Exception:
            z, Nz = None, 1
        is3d, k = (z is not None and Nz > 1), Nz // 2
        z_mid   = float(z[k]) if is3d else float("nan")
        t_code  = read_time(s)
        Bx = read_plane(s, F_BX, is3d, k, Ny, Nx)
        By = read_plane(s, F_BY, is3d, k, Ny, Nx)
        Jz = read_plane(s, F_JZ, is3d, k, Ny, Nx)
        Ez = read_plane(s, F_EZ, is3d, k, Ny, Nx)
        Nd = read_plane(s, F_N,  is3d, k, Ny, Nx)

    if Bx is None or Jz is None:
        continue

    #! average over X: kills PIC shot noise and keeps the Y structure
    dy_r, dx_r = spacing(y), spacing(x)
    Bx_y = np.mean(Bx, axis=1)

    records.append({"step":   step_from_fname(fname),
                    "t":      t_code,
                    "y":      np.asarray(y, dtype=float),
                    "dy":     dy_r,
                    "dx":     dx_r,
                    "Lx":     axis_span(x),
                    "zmid":   z_mid,
                    "ymid":   float(0.5 * ((y[0] - 0.5 * dy_r) + (y[-1] + 0.5 * dy_r))),
                    "Bx_y":   Bx_y,
                    "Jz_y":   np.mean(Jz, axis=1),
                    "curl_y": curl_z(Bx, By, dx_r, dy_r),
                    "divB":   div_b(Bx, By, dx_r, dy_r),
                    "Ez_y":   np.mean(Ez, axis=1) if Ez is not None else None,
                    "N_y":    np.mean(Nd, axis=1) if Nd is not None else None})

all_records = comm.gather(records, root=0)

if rank != 0:
    comm.Barrier()
    raise SystemExit

recs = [r for sub in all_records for r in sub]
if len(recs) == 0:
    print(f"No usable dumps in {base}", flush=True)
    comm.Barrier()
    raise SystemExit

recs.sort(key=lambda r: (r["t"] if np.isfinite(r["t"]) else r["step"]))

y, dy  = recs[0]["y"], recs[0]["dy"]
cs_y   = recs[0]["ymid"]                                   #! pgen puts the sheet at 0.5*(ymin + ymax)
Lx_run = recs[0]["Lx"]
have_E = all(r["Ez_y"] is not None for r in recs)
have_N = all(r["N_y"] is not None for r in recs)
in_win = np.abs(y - cs_y) <= args.ywin

have_scales = (args.larmor0 is not None and args.larmor0 > 0.0)
amp         = args.skindepth0 ** 2 / args.larmor0 if have_scales else 1.0
#! sigma0 = (skindepth0/larmor0)^2 is the per-unit-mass magnetisation set by [scales]. The TOTAL
#! magnetisation, the one that sets the outflow speed, divides by the inertia of both species:
#!   sigma_tot = 2 * sigma0 / (1 + mass_ratio)        (= sigma0 for pairs, ~sigma_i for ion-electron)
#! and v_A/c = sqrt(sigma_tot / (1 + sigma_tot)) normalises the reconnection rate, E_z / (B0 * v_A).
#! Assumes B_hat = 1, which the corrected pgen guarantees; B0 itself is measured from the t = 0 jump.
sigma0    = (args.skindepth0 / args.larmor0) ** 2 if have_scales else float("nan")
sigma_tot = 2.0 * sigma0 / (1.0 + args.mass_ratio) if have_scales else float("nan")
v_A       = np.sqrt(sigma_tot / (1.0 + sigma_tot)) if have_scales else float("nan")

#! ============================================================
#! Time series: scale, correlation, peak, thickness, E_z
#! ============================================================

t_arr, I_full, dBx, Jpk, Jhw, Ez0, d_scale, d_corr = [], [], [], [], [], [], [], []

for r in recs:
    Jy, Cy = r["Jz_y"], r["curl_y"]
    I_full.append(np.sum(Jy) * r["dy"])
    dBx.append(float(r["Bx_y"][-1] - r["Bx_y"][0]))
    Jpk.append(float(Jy[int(np.argmax(np.abs(Jy)))]))
    Jhw.append(half_width_half_max(Jy, y))
    Ez0.append(float(r["Ez_y"][int(np.argmin(np.abs(y - cs_y)))]) if r["Ez_y"] is not None else float("nan"))
    t_arr.append(r["t"] if np.isfinite(r["t"]) else float(r["step"]))
    #! fit inside the sheet window and only where curl B is significant
    m = in_win & (np.abs(Cy) > CURL_FRAC * np.max(np.abs(Cy[in_win])))
    s_fit, c_fit = scale_and_corr(Jy[m], Cy[m])
    d_scale.append(s_fit)
    d_corr.append(c_fit)

t_arr, I_full, dBx = map(np.asarray, (t_arr, I_full, dBx))
Jpk, Jhw, Ez0      = map(np.asarray, (Jpk, Jhw, Ez0))
d_scale, d_corr    = np.asarray(d_scale), np.asarray(d_corr)

t_plot = t_arr / Lx_run if (np.isfinite(Lx_run) and Lx_run > 0.0) else t_arr
t_lab  = rf"$t\,c/L_x$   ($L_x = {Lx_run:g}$)" if (np.isfinite(Lx_run) and Lx_run > 0.0) else r"$t$ [code time]"

#! ---- Ampere residual and the displacement current ----
C_map = np.array([r["curl_y"][in_win] for r in recs]).T
J_map = np.array([r["Jz_y"][in_win] for r in recs]).T
R_map = C_map - J_map / amp
y_win = y[in_win]
dt_dump = float(np.median(np.diff(t_arr))) if len(t_arr) > 1 else float("nan")

if have_E and len(recs) >= 3:
    E_map    = np.array([r["Ez_y"][in_win] for r in recs]).T
    dEdt_map = np.gradient(E_map, t_arr, axis=1)
else:
    E_map, dEdt_map = None, None

i_c   = int(np.argmin(np.abs(y_win - cs_y)))
R_cut = R_map[i_c, :]
E_cut = dEdt_map[i_c, :] if dEdt_map is not None else np.full_like(R_cut, np.nan)

#! ---- the written J against what the written density and the drift predict ----
if have_N:
    nf      = args.n_factor
    n_bg    = float(np.median(nf * recs[0]["N_y"][~in_win])) if np.any(~in_win) else np.nan
    B0_meas = 0.5 * abs(dBx[0])
    if args.beta_d is not None:
        beta_d = float(args.beta_d)
    elif args.cs_density is not None and args.cs_width is not None:
        #! each species carries n_cs/2 and they counter-drift, so J = n_cs * beta_d with n_cs the total
        beta_d = amp * B0_meas / (args.cs_density * args.cs_width)
    else:
        beta_d = float("nan")
    Jpred_pk, Jratio = [], []
    for r in recs:
        Jp = -(nf * np.asarray(r["N_y"], dtype=float) - n_bg) * beta_d
        Jpred_pk.append(float(Jp[int(np.argmax(np.abs(Jp)))]))
        Jratio.append(float(r["Jz_y"][int(np.argmax(np.abs(r["Jz_y"])))] / Jpred_pk[-1]) if Jpred_pk[-1] != 0.0 else float("nan"))
    Jpred_pk, Jratio = np.asarray(Jpred_pk), np.asarray(Jratio)
else:
    n_bg, beta_d = float("nan"), float("nan")
    Jpred_pk, Jratio = np.full(len(recs), np.nan), np.full(len(recs), np.nan)

#! ============================================================
#! Report: one line per test, no numbers unless they are actionable
#! ============================================================

lines = []

#! ---- t = 0 state: is the sheet in discrete equilibrium? ----
r0, C0 = recs[0], recs[0]["curl_y"]
J0     = recs[0]["Jz_y"] / amp
m0     = np.abs(J0) > 0.2 * np.max(np.abs(J0))
kappa  = float(np.mean(C0[m0] / J0[m0])) if np.count_nonzero(m0) >= 3 else float("nan")

if not np.isfinite(kappa):
    lines.append("t = 0 state   : cannot be judged, J_z is ~0 across the window")
elif abs(kappa - 1.0) < SCALE_TOL:
    lines.append("t = 0 state   : in discrete equilibrium")
elif abs(kappa - 1.0) < SMOOTH_TOL:
    lines.append("t = 0 state   : in equilibrium; minor mismatch between J_z and curl B from current smoothing")
else:
    lines.append(f"t = 0 state   : NOT in equilibrium, curl B / J = {kappa:.3g} -- the sheet will launch waves")

if np.isfinite(r0["divB"]) and r0["divB"] / (np.max(np.abs(C0)) + 1.0e-30) > DIVB_TOL:
    lines.append("                div B is not negligible -- check the field initialisation")

#! ---- the pointwise law, and whether structure develops ----
good = np.isfinite(d_scale) & np.isfinite(d_corr)
if np.count_nonzero(good) < 2:
    lines.append("pointwise law : too few dumps")
else:
    s_off = abs(float(np.median(d_scale[good])) / amp - 1.0)
    if s_off < SCALE_TOL:
        line = "pointwise law : J_z = AMP_COEFF * curl B holds"
    elif s_off < SMOOTH_TOL:
        line = "pointwise law : holds, with a minor peak deficit from current smoothing"
    else:
        line = f"pointwise law : scale is {float(np.median(d_scale[good])):.3g}, not AMP_COEFF = {amp:.3g}"
    below = np.where(d_corr < CORR_TOL)[0]
    if below.size > 0:
        line += f"; the profiles develop structure from t c/Lx = {t_plot[below[0]]:.2g}"
    lines.append(line)

#! ---- Ampere closure: only meaningful if the dumps resolve dE_z/dt ----
if dEdt_map is None:
    lines.append("Ampere closure: skipped, needs E in the dumps")
elif not np.isfinite(dt_dump) or dt_dump > DT_TOL:
    lines.append("Ampere closure: not testable, the dumps are too widely spaced for dE_z/dt")
else:
    m_e = np.abs(R_map) > CURL_FRAC * np.max(np.abs(R_map))
    with np.errstate(divide="ignore", invalid="ignore"):
        med_clos = float(np.nanmedian(np.abs(np.where(m_e, R_map / dEdt_map, np.nan))))
    if 0.5 <= med_clos <= 2.0:
        lines.append("Ampere closure: the residual is displacement current, as it should be")
    else:
        lines.append(f"Ampere closure: the residual is {med_clos:.3g}x the displacement current -- check the normalisations")

#! ---- the written J against the written density ----
if not have_N or not np.isfinite(Jratio[0]):
    lines.append("J vs density  : skipped, needs N in the dumps and --cs-density / --cs-width")
else:
    r_0   = Jratio[0]
    r_end = float(np.nanmedian(Jratio[max(1, int(0.9 * len(Jratio))):]))
    if abs(r_0 - 1.0) < SCALE_TOL:
        line = "J vs density  : the written J matches the density and the drift"
    elif abs(r_0 - 1.0) < SMOOTH_TOL:
        line = "J vs density  : consistent at t = 0; minor peak deficit from current smoothing"
    else:
        line = f"J vs density  : the written J is {r_0:.3g}x the prediction -- the two moments are not on the same normalisation"
    if np.isfinite(r_end) and abs(r_end / r_0 - 1.0) > 0.1:
        line += f"; the sheet drift relaxes to {100.0 * r_end / r_0:.0f}% of its initial value"
    lines.append(line)

header = f"\nHarris sheet diagnostics   |   {len(recs)} dumps   |   AMP_COEFF = {amp:.4g}"
header += f"   |   sigma = {sigma_tot:.4g}, v_A = {v_A:.4g}" if have_scales else "   [no --larmor0: comparing against the RAW J]"
print(header, flush=True)
for line in lines:
    print("    " + line, flush=True)

#! ============================================================
#! Figure:  (a) (b)
#!          (c) (e)
#!          (d) across both columns
#! ============================================================

fig, axd = plt.subplot_mosaic([["a", "b"], ["c", "e"], ["d", "d"]], figsize=(14.0, 15.0), constrained_layout=True)
#! ---- (a) the FIRST dump only: the deposited current against the field it has to support. curl B is
#! ---- rescaled by the best-fit scale, so this compares shapes; the two should lie on top of each other.
ax = axd["a"]
s0 = d_scale[0] if np.isfinite(d_scale[0]) else 1.0
ax.plot(y[in_win], recs[0]["Jz_y"][in_win], "-", color="crimson", lw=2.6, label=r"$\langle J_z\rangle_x$")
ax.plot(y[in_win], s0 * recs[0]["curl_y"][in_win], "--", color="navy", lw=1.8, dashes=(5, 3), label=rf"$s\,(\nabla\times B)_z$,  $s = {s0:.3g}$")
ax.axvline(cs_y, color="k", ls=":", lw=1.0)
ax.set_xlabel(r"$y\ \omega_p/c$", fontsize=12)
ax.set_ylabel("amplitude", fontsize=12)
ax.set_title(rf"(a) $J_z$ and $(\nabla\times B)_z$ at $t = {t_plot[0]:.3g}$", fontsize=13)
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

#! ---- (b) the fitted scale against AMP_COEFF: a pure code check, it must not drift
ax = axd["b"]
ax.plot(t_plot, d_scale, "o-", color="purple", lw=1.6, ms=4)
ax.axhline(amp, color="k", ls=":", lw=1.4, label=rf"AMP_COEFF = {amp:.4g}")
ax.set_xlabel(t_lab, fontsize=12)
ax.set_ylabel(r"best-fit $s$ in $J_z = s\,(\nabla\times B)_z$", fontsize=11)
ax.set_title(r"(b) normalisation check (must be $\sim$ AMP_COEFF)", fontsize=13)
ax.legend(fontsize=12)
ax.grid(alpha=0.3)

#! ---- (c) amplitude and thickness: rising peak at fixed HWHM is acceleration, falling peak at growing
#! ---- HWHM is spreading. Once islands form the x-average mixes O- and X-points and the HWHM loses meaning.
ax = axd["c"]
ax.plot(t_plot, np.abs(Jpk), "o-", color="crimson", lw=1.6, ms=4, label=r"$|J_z|_{\rm peak}$ written")
if have_N:
    ax.plot(t_plot, np.abs(Jpred_pk), "s--", color="darkgreen", lw=1.4, ms=4, label=r"$|n_{\rm cs}\beta_d|_{\rm peak}$ from $n$")
ax.set_xlabel(t_lab, fontsize=12)
ax.set_ylabel("peak current", fontsize=12)
ax.tick_params(axis="y", labelcolor="crimson")
ax.legend(fontsize=14, loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2, frameon=False)
ax.grid(alpha=0.3)
axb = ax.twinx()
axb.plot(t_plot, Jhw, "^:", color="navy", lw=1.4, ms=4)
axb.set_ylabel(r"$J_z$ layer HWHM [code units]", color="navy", fontsize=12)
axb.tick_params(axis="y", labelcolor="navy")
if args.cs_width is not None:
    axb.axhline(0.8814 * args.cs_width, color="navy", ls=":", lw=1.2)   #! sech^2 HWHM = arccosh(sqrt(2)) * cs_width


#! ---- (d) the reconnection field across the sheet
ax = axd["d"]
if have_E:
    Ez_map = np.array([r["Ez_y"][in_win] for r in recs]).T
    vmax   = max(np.percentile(np.abs(Ez_map), 99.0), 1.0e-30)
    T, Y   = np.meshgrid(t_plot, y_win)                        #! pcolormesh: the time axis is not equidistant
    im = ax.pcolormesh(T, Y, Ez_map, cmap="seismic", vmin=-vmax, vmax=vmax, shading="nearest")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.axhline(cs_y, color="k", ls=":", lw=1.0)
    ax.set_ylabel(r"$y\ \omega_p/c$", fontsize=12)
else:
    ax.text(0.5, 0.5, f"{F_EZ} not in the dumps", ha="center", va="center", transform=ax.transAxes, fontsize=12)
ax.set_xlabel(t_lab, fontsize=12)
ax.set_title(r"(d) $E_z(y,t)$ across the sheet", fontsize=13)

#! ---- (e) the reconnection rate: |E_z| at the X-line divided by B0 * v_A, with B0 measured at t = 0.
#! ---- The usual relativistic-pair answer is ~0.1.
#! ---- CAVEATS: B0 is the INITIAL upstream field, so once it decays the true rate is higher than plotted;
#! ---- and the x-average mixes X-points with O-points, so the value is a lower bound once islands form.
#! ---- The rise below the first ~0.2 light-crossing times is the start-up transient, not reconnection.
ax = axd["e"]
B0_norm = 0.5 * abs(dBx[0])
if have_E and have_scales and B0_norm > 0.0:
    #! the sign of E_z is set by the current direction, not by the physics, so the rate is |E_z|
    ax.plot(t_plot, np.abs(Ez0) / (B0_norm * v_A), "o-", color="darkgreen", lw=1.6, ms=4)
    ax.set_ylabel(r"$|E_z| / (B_0 v_A)$", fontsize=12)
    zlab = f" at $z = {recs[0]['zmid']:.3g}$" if np.isfinite(recs[0]["zmid"]) else ""
    ax.set_title(rf"(e) $R_{{\rm rate}}${zlab}   ($\sigma = {sigma_tot:.3g}$, $v_A = {v_A:.3g}$)", fontsize=13)
elif have_E:
    ax.plot(t_plot, Ez0, "o-", color="darkgreen", lw=1.6, ms=4)
    ax.set_ylabel(r"$E_z(y=y_0)$", fontsize=12)
    ax.set_title("(e) reconnection field (pass --larmor0 to normalise)", fontsize=13)
else:
    ax.text(0.5, 0.5, "no E data", ha="center", va="center", transform=ax.transAxes, fontsize=12)
    ax.set_title(r"(e) $R_{\rm rate}$", fontsize=13)
ax.axhline(0.0, color="k", lw=0.8)
ax.set_xlabel(t_lab, fontsize=12)
ax.grid(alpha=0.3)

fig.suptitle("Harris sheet diagnostics", fontsize=14)
outfile = f"{outdir}/CS_transient_check.png"
fig.savefig(outfile, dpi=150, bbox_inches="tight")
plt.close(fig)

#! ============================================================
#! CSV of the time series, so the numbers can be re-plotted
#! ============================================================

csv   = f"{outdir}/CS_transient_check.csv"
hdr   = "step,t_code,int_Jz,dBx,Jz_peak,Jz_HWHM,Ez_centre,curl_scale,curl_corr,R_centre,dEzdt_centre,Jz_peak_pred,Jz_peak_ratio"
steps = np.array([r["step"] for r in recs], dtype=float)
np.savetxt(csv, np.column_stack([steps, t_arr, I_full, dBx, Jpk, Jhw, Ez0, d_scale, d_corr, R_cut, E_cut, Jpred_pk, Jratio]), delimiter=",", header=hdr, comments="")

print(f"\n    Saved {outfile}\n    Saved {csv}\n", flush=True)

comm.Barrier()