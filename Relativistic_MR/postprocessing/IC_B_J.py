"""
t=0 initial-condition diagnostics for the Entity relativistic Harris sheet.

PRIMARY QUESTION: is the sheet in discrete equilibrium, or does it launch the
startup rings? At a true t=0 equilibrium the Maxwell update dE/dt = curl(B) - J
vanishes, so the residual  r = (curl B)_z - J_z  IS the launch source.

Usage
-----
    srun -N 1 -n 1 python3 -u ../postprocessing/IC_B_J.py "$fields" [outdir] \
         [--cs_density 10] [--cs_width 1.0]

    $fields   : directory holding fields.*.bp
    outdir    : where to save PNGs (default: same as $fields)
    --cs_density : the setup.cs_density you INPUT (enables the localizer, #3)
    --cs_width   : the setup.cs_width you INPUT (enables a resolution line)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from adios2 import Stream
import argparse, os, glob

#! ============================================================
#! Args
#! ============================================================
ap = argparse.ArgumentParser()
ap.add_argument("base", help="directory with fields.*.bp")
ap.add_argument("outdir", nargs="?", default=None, help="output dir (default: base)")
ap.add_argument("--cs_density", type=float, default=None,
                help="input setup.cs_density; enables the drift-vs-density localizer")
ap.add_argument("--cs_width", type=float, default=None,
                help="input setup.cs_width [d0]; enables a resolution line")
args   = ap.parse_args()
base   = args.base
outdir = args.outdir if args.outdir is not None else base
os.makedirs(outdir, exist_ok=True)

TIME_KEY = "Time"

#! ============================================================
#! Locate earliest dump (smallest step index)
#! ============================================================
files = sorted(glob.glob(f"{base}/fields.*.bp"), key=lambda f: int(os.path.basename(f).rsplit(".", 1)[0].split(".")[-1]))
if not files:
    raise SystemExit(f"No fields.*.bp in {base}")
fname = files[0]

#! ============================================================
#! Robust readers
#! ============================================================
def read_var(s, name):
    #! return array or None if absent
    try:
        return np.asarray(s.read(name))
    except Exception:
        return None

def read_time(s):
    for getter in (lambda: s.read(TIME_KEY), lambda: s.read_attribute(TIME_KEY)):
        try:
            v = getter()
            if v is not None:
                return float(np.asarray(v).ravel()[0])
        except Exception:
            pass
    return float("nan")

def avail(s):
    try:
        return set(s.available_variables().keys())
    except Exception:
        return set()

#! ============================================================
#! Read everything we might need from the ONE earliest file
#! ============================================================
with Stream(fname, "r") as s:
    next(s.steps())
    names = avail(s)
    x  = read_var(s, "X1").ravel()
    y  = read_var(s, "X2").ravel()
    B1 = read_var(s, "fB1")                 #! B_x (reversing field)
    B2 = read_var(s, "fB2")                 #! B_y
    B3 = read_var(s, "fB3")                 #! B_z (guide)
    E1 = read_var(s, "fE1")
    E2 = read_var(s, "fE2")
    E3 = read_var(s, "fE3")                 #! E_z: driven by the (curl B - J)_z residual
    J3 = read_var(s, "fJ3")                 #! J_z (out-of-plane current)
    t_code = read_time(s)

    #! density: prefer total fN; else sum per-species fN_1, fN_2, ...
    N = read_var(s, "fN")
    if N is None:
        sp = sorted(n for n in names if n.startswith("fN_"))
        if sp:
            N = sum(np.asarray(s.read(n)) for n in sp)
            print(f"  [density] summed per-species: {sp}")
    if N is None:
        print("  [density] WARNING: no fN or fN_* found; density checks skipped")

dx = float(np.mean(np.diff(x)))
dy = float(np.mean(np.diff(y)))
Lx = float(x.max() - x.min())

#! ============================================================
#! 1. AMPERE BALANCE
#!    (curl B)_z = dB2/dx - dB1/dy   [arrays (Ny,Nx): axis0=y, axis1=x]
#! ============================================================
curlB_z = np.gradient(B2, dx, axis=1) - np.gradient(B1, dy, axis=0)
curl_y  = curlB_z.mean(axis=1)             #! x-average -> 1D in y
J_y     = J3.mean(axis=1)

Jpk  = np.max(np.abs(J_y))
mask = np.abs(J_y) > 0.2 * Jpk             #! sheet region only (skip noise floor)
if mask.sum() < 3:
    raise SystemExit("J_z ~0 everywhere -- wrong variable name or empty sheet?")

ratio      = curl_y[mask] / J_y[mask]
kappa      = float(np.mean(ratio))
kappa_std  = float(np.std(ratio))
resid_y    = curl_y - J_y
resid_1    = float(np.max(np.abs(resid_y)) / Jpk)   #! residual assuming coeff=1

print("1. AMPERE BALANCE  (curl B)_z vs J_z")
print(f"     peak |J_z|            = {Jpk:.4g}")
print(f"     peak |(curl B)_z|     = {np.max(np.abs(curl_y)):.4g}")
print(f"     ratio curl/J          = {kappa:.4g} +/- {kappa_std:.2g}"
      f"  (spread {kappa_std/abs(kappa):.1%})")
print(f"     residual |curl-J|/|J| = {resid_1:.1%}  (coeff=1)")
if abs(kappa - 1.0) < 0.05:
    print("     => balanced (coeff 1): sheet should NOT launch waves")
elif kappa_std/abs(kappa) < 0.05:
    print(f"     => FLAT ratio {kappa:.3g} != 1: coherent residual drives E -> LAUNCH")
    print(f"        deposited current is 1/{kappa:.3g} of what curl(B) needs")
else:
    print("     => STRUCTURED residual (shape mismatch): check resolution/staggering")
print()

#! ============================================================
#! 2. DENSITY overdensity from fN
#! ============================================================
measured_over = None
if N is not None:
    prof = N.mean(axis=1)                                   #! 1D density in y
    edge = max(3, len(prof)//8)
    bg   = float(np.median(np.concatenate([prof[:edge], prof[-edge:]])))  #! both walls
    pk   = float(prof.max())
    measured_over = pk/bg - 1.0 if bg > 0 else float("nan")
    print("2. DENSITY  (from fN)")
    print(f"     background n_bg      = {bg:.4g}")
    print(f"     sheet peak           = {pk:.4g}")
    print(f"     measured overdensity = {measured_over:.4g}"
          + (f"   (input was {args.cs_density:.4g})" if args.cs_density else ""))
    print()

#! ============================================================
#! 3. LOCALIZER: is the factor in the DENSITY or the DRIFT?
#!    If drift is correct, curl/J = n_required/n_deposited = input_over/measured_over.
#! ============================================================
if (args.cs_density is not None) and (measured_over is not None) and measured_over > 0:
    dens_ratio = args.cs_density / measured_over
    print("3. LOCALIZER  (drift vs density)")
    print(f"     Ampere ratio curl/J            = {kappa:.4g}")
    print(f"     density ratio input/measured   = {dens_ratio:.4g}")
    if abs(kappa - dens_ratio) / max(kappa, 1e-30) < 0.1:
        print("     => MATCH: the deficit is entirely in the DEPOSITED DENSITY.")
        print("        Drift is fine; fix how cs_density / n0 maps to the injector.")
    else:
        print("     => MISMATCH: density alone doesn't explain it -> DRIFT or J")
        print("        normalisation is also off. Compare fJ3 to 2*fN*beta_d by hand.")
    print()

#! ============================================================
#! 4. div B  (field-init sanity; should be ~0)
#! ============================================================
divB = np.gradient(B1, dx, axis=1) + np.gradient(B2, dy, axis=0)
if B3 is not None and B3.ndim == 3:
    pass  #! 3D z-derivative would go here; these dumps are 2D
scaleB = np.max(np.abs(curl_y)) + 1e-30
print("4. div B  (should be ~0)")
print(f"     max|div B| / max|curl B| = {np.max(np.abs(divB))/scaleB:.2e}")
print()

#! ============================================================
#! 5. E-FIELD: the launch caught in the act
#!    E_z is driven by (curl B - J)_z, so its y-profile should mirror resid_y.
#! ============================================================
if E3 is not None:
    Ez_y = E3.mean(axis=1)
    e_pk = float(np.max(np.abs(Ez_y)))
    print("5. E-FIELD")
    for lbl, arr in (("E_x", E1), ("E_y", E2), ("E_z", E3)):
        if arr is not None:
            print(f"     peak |{lbl}| = {np.max(np.abs(arr)):.3g}")
    #! shape correlation between E_z and the Ampere residual
    if e_pk > 0 and np.max(np.abs(resid_y)) > 0:
        cc = float(np.corrcoef(Ez_y, resid_y)[0, 1])
        print(f"     corr(E_z profile, curl-J profile) = {cc:+.3f}"
              "   (near +/-1 confirms residual drives E_z)")
    print()

#! ============================================================
#! 6. B_y, B_z sanity at t=0
#! ============================================================
print("6. FIELD SANITY at t=0")
print(f"     peak |B_x| = {np.max(np.abs(B1)):.4g}   (= B_BG expected)")
print(f"     peak |B_y| = {np.max(np.abs(B2)):.3g}   (~0 expected)")
if B3 is not None:
    print(f"     peak |B_z| = {np.max(np.abs(B3)):.3g}   (= guide field expected)")
print()

#! ============================================================
#! PLOTS
#! ============================================================
#! Fig 1: the headline Ampere overlay
fig1, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
ax.plot(y, J_y,    lw=2.0, label=r"$J_z$ (deposited)")
ax.plot(y, curl_y, lw=1.5, ls="--", label=r"$(\nabla\times B)_z=-\partial_y B_x$")
ax.set_xlabel(r"$y\ [d_0]$", fontsize=13)
ax.set_ylabel("x-averaged amplitude", fontsize=13)
ax.set_title(f"Ampere balance at t=0   (curl/J = {kappa:.3g})", fontsize=13)
ax.legend(fontsize=11)
out1 = os.path.join(outdir, "ampere_check.png")
fig1.savefig(out1, dpi=150, bbox_inches="tight")

#! Fig 2: supporting diagnostics
fig2, axs = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)

axs[0, 0].plot(y, resid_y, color="crimson", lw=1.8)
axs[0, 0].axhline(0, color="k", lw=0.6)
axs[0, 0].set_title(r"Ampere residual $(\nabla\times B)_z - J_z$  (launch source)")
axs[0, 0].set_xlabel(r"$y\ [d_0]$"); axs[0, 0].set_ylabel("residual")

if N is not None:
    axs[0, 1].plot(y, prof, color="darkorange", lw=1.8)
    axs[0, 1].axhline(bg, color="gray", ls=":", label=f"n_bg={bg:.3g}")
    axs[0, 1].set_title(f"Density  (overdensity = {measured_over:.3g})")
    axs[0, 1].set_xlabel(r"$y\ [d_0]$"); axs[0, 1].set_ylabel(r"$n$")
    axs[0, 1].legend(fontsize=9)
else:
    axs[0, 1].set_visible(False)

if E3 is not None:
    axs[1, 0].plot(y, Ez_y, color="teal", lw=1.8)
    axs[1, 0].set_title(r"$E_z$ profile (should mirror residual)")
    axs[1, 0].set_xlabel(r"$y\ [d_0]$"); axs[1, 0].set_ylabel(r"$E_z$")
else:
    axs[1, 0].set_visible(False)

#! magnetic pressure profile B^2/2 (the part of pressure balance we CAN see;
#! particle pressure isn't in an N/B/E/J dump, so this is a partial check only)
B2sq = (B1**2 + B2**2 + (B3**2 if B3 is not None else 0.0)).mean(axis=1)
axs[1, 1].plot(y, 0.5 * B2sq, color="navy", lw=1.8)
axs[1, 1].set_title(r"Magnetic pressure $B^2/2$  (particle P not in this dump)")
axs[1, 1].set_xlabel(r"$y\ [d_0]$"); axs[1, 1].set_ylabel(r"$B^2/2$")

out2 = os.path.join(outdir, "ic_diagnostics.png")
fig2.savefig(out2, dpi=150, bbox_inches="tight")

print(f"Saved {out1}")
print(f"Saved {out2}\n\n")