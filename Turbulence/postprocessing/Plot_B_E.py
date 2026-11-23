"""
Created on Tue Nov 03 2026

@author: Pranab JD, Claude AI

Decaying kinetic turbulence -- B and E snapshots.

Usage:

    input="/scratch/project_465002528/pjd/turbulence/fields/"
    output="/scratch/project_465002528/pjd/turbulence/plots/"

    srun python3 ../postprocessing/Plot_B_E.py "$input" "$output" --Lx 256.0 --Ly 256.0 --larmor0 10.0 --u-rms 0.04

    Required:
        --Lx        box length along x in code units (d_e)
        --Ly        box length along y in code units (d_e)
        --Lz        box length along z in code units (d_e); REQUIRED for a 3D run,
                    rejected for a 2D run
        --larmor0   rho0 = m0 c^2 / (e B0) in code units, for Omega_ce t = t B_guide / rho0

    Optional:
        --b-guide   guide field [B0] for Omega_ce (default: 1)
        --u-rms     initial turbulent u_rms [c]; gives t_eddy = 1 / (k_min u_rms), k_min = 2 pi / L_x
        --t-eddy    outer-scale eddy turnover time [1/w_pe]; overrides --u-rms
        --pct       percentile used for the colour limits (default: 99.5)
        --per-frame rescale colours independently for every snapshot. The DEFAULT is
                    the opposite: one fixed scale shared by all frames, which is what
                    you want for a movie of a DECAYING run (an amplitude that halves
                    must look halved, not get renormalised away)

2D/3D compatibility
-------------------
    Entity writes field arrays with the SLOWEST axis first:
        2D run : (Ny, Nx)
        3D run : (Nz, Ny, Nx)        [axis0 = z, axis1 = y, axis2 = x]
        
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
from mpi4py import MPI
from adios2 import Stream
import argparse, os, glob
import matplotlib.pyplot as plt

#! ============================================================
#! Settings
#! ============================================================
TIME_KEY = "Time"
N_TICKS  = 5          #! exactly this many equidistant ticks on BOTH axes of EVERY panel
N_SAMPLE = 8          #! snapshots sampled when building the shared colour scale
BLOCK_DE = 1.0        #! maps are block-averaged over BLOCK_DE x BLOCK_DE [d_e] before plotting (no aliasing)

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
parser.add_argument("base",        type=str,   help="Directory with fields.NNNNNNNNN.bp files")
parser.add_argument("outdir",      type=str,   help="Output directory for PNG plots")
parser.add_argument("--Lx",        type=float, required=True, help="Box length along x in code units")
parser.add_argument("--Ly",        type=float, required=True, help="Box length along y in code units")
parser.add_argument("--Lz",        type=float, default=None,  help="Box length along z in code units (required for 3D runs)")
parser.add_argument("--larmor0",   type=float, required=True, help="rho0 = m0 c^2 / (e B0) in code units, for Omega_ce = B_guide / rho0")
parser.add_argument("--b-guide",   type=float, default=1.0,   dest="b_guide", help="Guide field [B0] for Omega_ce (default: 1)")
parser.add_argument("--u-rms",     type=float, default=None,  dest="u_rms", help="Initial turbulent u_rms [c]; t_eddy = 1 / (k_min u_rms), k_min = 2 pi / L_x")
parser.add_argument("--t-eddy",    type=float, default=None,  dest="t_eddy", help="Outer eddy turnover time [1/w_pe]; overrides --u-rms")
parser.add_argument("--pct",       type=float, default=99.5,  help="Percentile for colour limits")
parser.add_argument("--per-frame", action="store_true", dest="per_frame", help="Rescale colours per snapshot instead of using one fixed scale")

args      = parser.parse_args()
base      = args.base
outdir    = args.outdir
Lx        = args.Lx
Ly        = args.Ly
Lz        = args.Lz
t_eddy    = args.t_eddy
larmor0   = args.larmor0
b_guide   = args.b_guide
u_rms     = args.u_rms
pct       = args.pct
per_frame = args.per_frame

if Lx <= 0.0:
    parser.error("--Lx must be positive")
if Ly <= 0.0:
    parser.error("--Ly must be positive")
if Lz is not None and Lz <= 0.0:
    parser.error("--Lz must be positive")
if not (50.0 < pct <= 100.0):
    parser.error("--pct must be in (50, 100]")
if t_eddy is not None and t_eddy <= 0.0:
    parser.error("--t-eddy must be positive")
if larmor0 <= 0.0:
    parser.error("--larmor0 must be positive")
if u_rms is not None and u_rms <= 0.0:
    parser.error("--u-rms must be positive")

#! outer-scale eddy turnover time: t_eddy = 1 / (k_min u_rms),  k_min = 2 pi / L_x
if t_eddy is None and u_rms is not None:
    t_eddy = Lx / (2.0 * np.pi * u_rms)

def time_label_text(t_code):
    #! t c/L_x,  Omega_ce t = t B_guide / rho0,  omega_pe t = t (code time unit = 1/omega_pe, total density),  t / t_eddy
    label = rf"$t$ = {t_code / Lx:.2f} $L_x/c$,      $\Omega_{{c,e}} t$ = {t_code * abs(b_guide) / larmor0:.1f},      $\omega_{{p,e}} t$ = {t_code:.1f}"
    if t_eddy is not None:
        label += rf",      $t/\tau_{{\rm eddy}}$ = {t_code / t_eddy:.2f}"
    return label

if rank == 0:
    os.makedirs(outdir, exist_ok=True)
comm.Barrier()

#! ============================================================
#! Panel definition
#!   label -> (kind, cmap, group)
#!   kind  = "div" : diverging, symmetric limits about 0 (signed components)
#!   kind  = "seq" : sequential, limits [0, vmax]        (magnitudes)
#!   group : panels sharing a group share one colour scale
#! ============================================================
PANELS = [
    (r"$B_x$",        "div", "seismic", "B_inplane"),
    (r"$B_y$",        "div", "seismic", "B_inplane"),
    (r"$B_z$",        "seq", "plasma",  "B_z"),
    (r"$|B_{XY}|$",   "seq", "inferno", "B_mag"),
    (r"$E_x$",        "div", "seismic", "E_inplane"),
    (r"$E_y$",        "div", "seismic", "E_inplane"),
    (r"$E_z$",        "div", "seismic", "E_z"),
    (r"$|E_{XY}|$",   "seq", "inferno", "E_mag"),
]

PANEL_UNITS = {
    r"$B_x$":       r"[$B_0$]",
    r"$B_y$":       r"[$B_0$]",
    r"$B_z$":       r"[$B_0$]",
    r"$|B_{XY}|$":  r"[$B_0$]",
    r"$E_x$":       r"[$B_0$]",
    r"$E_y$":       r"[$B_0$]",
    r"$E_z$":       r"[$B_0$]",
    r"$|E_{XY}|$":  r"[$B_0$]",
}

GROUP_KIND = {}
for _label, _kind, _cmap, _group in PANELS:
    GROUP_KIND[_group] = _kind

#! ============================================================
#! Helpers
#! ============================================================
files = sorted(glob.glob(f"{base}/fields.*.bp"))

if len(files) == 0:
    if rank == 0:
        print(f"ERROR: no fields.*.bp in {base}", flush=True)
    raise SystemExit(1)

def step_from_fname(fname):
    base_name = os.path.basename(fname)          # fields.000000251.bp
    stem      = base_name.rsplit(".", 1)[0]      # fields.000000251
    try:
        return int(stem.split(".")[-1])          # 251
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

def grid_sizes(stream):
    """
    (Nx, Ny, Nz) from the coordinate arrays; Nz is None for a 2D dump. The coordinate
    arrays are 1D and tiny, so reading them whole costs nothing and settles the
    dimensionality without touching a field variable.
    """
    Nx = np.asarray(stream.read("X1")).size
    Ny = np.asarray(stream.read("X2")).size
    try:
        Nz = np.asarray(stream.read("X3")).size
    except Exception:
        Nz = None
    return Nx, Ny, (Nz if (Nz is not None and Nz > 1) else None)

def probe_dimensionality(fname):
    with Stream(fname, "r") as s:
        next(s.steps())
        _, _, Nz = grid_sizes(s)
    return (Nz is not None), (int(Nz) if Nz is not None else 0)

is_3d, Nz_global = (None, None)
if rank == 0:
    is_3d, Nz_global = probe_dimensionality(files[0])
is_3d, Nz_global = comm.bcast((is_3d, Nz_global), root=0)

if is_3d and Lz is None:
    if rank == 0:
        print("ERROR: this is a 3D run (X3 present); --Lz is required", flush=True)
    raise SystemExit(1)
if (not is_3d) and Lz is not None:
    if rank == 0:
        print("ERROR: this is a 2D run; --Lz is not applicable", flush=True)
    raise SystemExit(1)

if rank == 0:
    print(f"Found {len(files)} files", flush=True)
    print(f"    Dimensionality      = {'3D' if is_3d else '2D'}", flush=True)
    print(f"    Box length L_x [d_e] = {Lx}", flush=True)
    print(f"    Box length L_y [d_e] = {Ly}", flush=True)
    if is_3d:
        print(f"    Box length L_z [d_e] = {Lz}", flush=True)
        print(f"    Cut plane           = z = {0.5 * Lz:g} (k = {Nz_global // 2} of {Nz_global})", flush=True)
    print(f"    rho0, B_guide        = {larmor0:g}, {b_guide:g}", flush=True)
    if t_eddy is not None:
        print(f"    Eddy turnover [1/w_pe] = {t_eddy:.6g}", flush=True)
    else:
        print("    Eddy turnover        = not given (pass --u-rms or --t-eddy)", flush=True)
    print(f"    Colour scale        = {'per frame' if per_frame else 'fixed across all frames'}", flush=True)
    print(" ", flush=True)

files_local = files[rank::size]

def read_plane(stream, name, Nx, Ny, Nz, k, missing):
    """
    One x-y plane as (Ny, Nx): rows = y, columns = x, which is what imshow expects
    with origin='lower' and extent=[0, Lx, 0, Ly].
      2D: the whole (Ny, Nx) array.
      3D: ONLY the k-th plane, as a hyperslab -- start/count, not a numpy slice.
    Returns None, and records the name in 'missing', if the variable is absent or has
    the wrong plane size.
    """
    try:
        if Nz is None:
            a = np.asarray(stream.read(name))
        else:
            a = np.asarray(stream.read(name, start=[int(k), 0, 0], count=[1, int(Ny), int(Nx)]))
    except Exception:
        missing.add(name)
        return None

    if a is None or a.size != int(Nx) * int(Ny):
        missing.add(name)
        return None

    return a.reshape(int(Ny), int(Nx))

def block_size(Nx, Ny):
    #! divisor of gcd(Nx, Ny) closest to BLOCK_DE in cells, so the blocks tile the plane exactly
    target   = BLOCK_DE * Nx / Lx
    common   = int(np.gcd(int(Nx), int(Ny)))
    divisors = [d for d in range(1, common + 1) if common % d == 0]
    return min(divisors, key=lambda d: abs(d - target))

def block_average(data, block):
    #! mean over block x block cells, ignoring blanked (NaN) cells
    if data is None or block <= 1:
        return data
    rows, columns = data.shape
    blocks = data.reshape(rows // block, block, columns // block, block)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.nansum(blocks, axis=(1, 3)) / np.sum(np.isfinite(blocks), axis=(1, 3))

def load_panels(fname):
    """
    Everything one figure needs, from one open of the file. Used BOTH by the colour-
    scale scan and by the plotting loop, so the two can never disagree about what a
    panel contains.

    Returns (panels, t_code, x, y, missing); a panel value of None means "not in the
    dump, leave it blank".
    """
    missing = set()

    with Stream(fname, "r") as s:
        next(s.steps())

        x      = np.asarray(s.read("X1"))
        y      = np.asarray(s.read("X2"))
        t_code = read_time(s)

        Nx, Ny, Nz = grid_sizes(s)
        k_mid = (Nz // 2) if Nz is not None else 0      #! mid-plane in z, 3D only

        Bx = read_plane(s, "fB1", Nx, Ny, Nz, k_mid, missing)
        By = read_plane(s, "fB2", Nx, Ny, Nz, k_mid, missing)
        Bz = read_plane(s, "fB3", Nx, Ny, Nz, k_mid, missing)

        Ex = read_plane(s, "fE1", Nx, Ny, Nz, k_mid, missing)
        Ey = read_plane(s, "fE2", Nx, Ny, Nz, k_mid, missing)
        Ez = read_plane(s, "fE3", Nx, Ny, Nz, k_mid, missing)

    have_B = (Bx is not None) and (By is not None) and (Bz is not None)
    have_E = (Ex is not None) and (Ey is not None) and (Ez is not None)

    #! in-plane magnitudes: |B_XY| = sqrt(Bx^2 + By^2), |E_XY| = sqrt(Ex^2 + Ey^2)
    B_mag = np.sqrt(Bx**2 + By**2) if have_B else None
    E_mag = np.sqrt(Ex**2 + Ey**2) if have_E else None

    block = block_size(Nx, Ny)
    panels = {
        r"$B_x$":       block_average(Bx, block),
        r"$B_y$":       block_average(By, block),
        r"$B_z$":       block_average(Bz, block),
        r"$|B_{XY}|$":  block_average(B_mag, block),
        r"$E_x$":       block_average(Ex, block),
        r"$E_y$":       block_average(Ey, block),
        r"$E_z$":       block_average(Ez, block),
        r"$|E_{XY}|$":  block_average(E_mag, block),
    }
    return panels, t_code, x, y, missing

def panel_extremum(data, kind):
    if data is None:
        return None
    v = np.percentile(np.abs(data), pct) if kind == "div" else np.percentile(data, pct)
    return float(v)

def limits_from_extrema(acc):
    limits = {}
    for group, kind in GROUP_KIND.items():
        v = acc[group]
        if v is None or not np.isfinite(v) or v <= 0.0:
            v = 1.0
        limits[group] = (-v, v) if kind == "div" else (0.0, v)
    return limits

def accumulate(acc, panels):
    for label, kind, _, group in PANELS:
        v = panel_extremum(panels[label], kind)
        if v is None:
            continue
        acc[group] = v if acc[group] is None else max(acc[group], v)
    return acc

def scan_limits(all_files):
    """
    One colour scale for the whole movie, built on rank 0 from N_SAMPLE snapshots
    spread evenly over the run, then broadcast.

    Sampling rather than using the first file matters here: the pgen starts with E = 0,
    so limits taken from t = 0 would make every E panel saturate later. Sampling
    rather than ALL files keeps the cost at a few extra reads.
    """
    limits = None
    if rank == 0:
        idx = np.unique(np.linspace(0, len(all_files) - 1, min(N_SAMPLE, len(all_files))).astype(int))
        acc = {group: None for group in GROUP_KIND}
        missing = set()
        for i in idx:
            panels, _, _, _, miss = load_panels(all_files[i])
            missing |= miss
            accumulate(acc, panels)

        limits = limits_from_extrema(acc)

        if missing:
            print(f"  [warn] not in the dump, panels left blank: {', '.join(sorted(missing))}", flush=True)
        print(f"Fixed colour limits from {len(idx)} sampled snapshots (p{pct}):", flush=True)
        for group in GROUP_KIND:
            tag = "" if acc[group] is not None else "   (no data)"
            print(f"    {group:12s} [{limits[group][0]:+.4g}, {limits[group][1]:+.4g}]{tag}", flush=True)
        print(" ", flush=True)

    return comm.bcast(limits, root=0)

def frame_limits(panels):
    acc = {group: None for group in GROUP_KIND}
    accumulate(acc, panels)
    return limits_from_extrema(acc)

def set_equidistant_ticks(ax, xlo, xhi, ylo, yhi, n=N_TICKS):
    xt = np.linspace(xlo, xhi, n)
    yt = np.linspace(ylo, yhi, n)
    ax.set_xticks(xt)
    ax.set_yticks(yt)
    ax.set_xticklabels([f"{v:g}" for v in xt])
    ax.set_yticklabels([f"{v:g}" for v in yt])
    ax.minorticks_off()                     #! "no more, no less" -> no minor ticks either

#! ============================================================
#! Shared colour scale (unless --per-frame)
#! ============================================================
fixed_limits = None if per_frame else scan_limits(files)
comm.Barrier()

#! ============================================================
#! Loop over assigned files
#! ============================================================
x_lo, x_hi = 0.0, Lx
y_lo, y_hi = 0.0, Ly

for fname in files_local:

    step_idx = step_from_fname(fname)

    panels, t_code, x, y, missing = load_panels(fname)

    #! sanity check: the file should agree with --Lx/--Ly to within about one cell
    if rank == 0 and fname == files_local[0]:
        dx_est = (x.max() - x.min()) / max(x.size - 1, 1)
        dy_est = (y.max() - y.min()) / max(y.size - 1, 1)
        if abs(x.max() - x.min() + dx_est - Lx) > 2.0 * dx_est or \
           abs(y.max() - y.min() + dy_est - Ly) > 2.0 * dy_est:
            print(f"  [warn] file extent is x:[{x.min():g},{x.max():g}] y:[{y.min():g},{y.max():g}] "
                  f"but --Lx={Lx:g} --Ly={Ly:g}", flush=True)

    limits = fixed_limits if fixed_limits is not None else frame_limits(panels)

    #! ========================================================
    #! Time labels. t_code is t w_p; t c/L_x is the light crossing count.
    #! ========================================================
    if not np.isnan(t_code):
        time_label = time_label_text(t_code)
        if is_3d:
            time_label += f"   |   $z = {0.5 * Lz:g}\\,d_e$"
    else:
        time_label = f"step {step_idx:09d}   (time not found -- check TIME_KEY)"

    #! ========================================================
    #! Plot
    #! ========================================================
    fig, axs = plt.subplots(2, 4, figsize=(16, 8))

    for ax, (label, kind, cmap, group) in zip(axs.flat, PANELS):

        data = panels[label]

        ax.set_title(label)
        ax.tick_params(axis="both", which="major", labelsize=10, length=6)
        ax.set_xlabel(r"$x\ [d_e]$")
        ax.set_ylabel(r"$y\ [d_e]$")
        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(y_lo, y_hi)
        ax.set_aspect("equal")

        #! exactly 5 equidistant ticks + labels on each axis of this panel
        set_equidistant_ticks(ax, x_lo, x_hi, y_lo, y_hi)

        if data is None:
            ax.text(0.5, 0.5, "not in dump", transform=ax.transAxes,
                    ha="center", va="center", fontsize=11, color="0.5")
            continue

        vmin, vmax = limits[group]

        im = ax.imshow(data, origin="lower", aspect="equal",
                       extent=[x_lo, x_hi, y_lo, y_hi], cmap=cmap, vmin=vmin, vmax=vmax)

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        if PANEL_UNITS[label]:
            cbar.set_label(PANEL_UNITS[label])

    fig.suptitle(time_label, fontsize=16, y=1.00)
    fig.tight_layout()

    outfile = f"{outdir}/fields_2D_{step_idx:09d}.png"
    fig.savefig(outfile, dpi=150)
    plt.close(fig)

    print(f"Saved {outfile}", flush=True)

comm.Barrier()

if rank == 0:
    print("\nDone.", flush=True)
    print("\n--------------------------\n\n", flush=True)