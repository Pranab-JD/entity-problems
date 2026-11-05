"""
Created on Tue Nov 03 2026

@author: Pranab JD, Claude AI

Decaying kinetic turbulence -- B and E snapshots.

Usage:

    input="/scratch/project_465002528/pjd/turbulence/fields/"
    output="/scratch/project_465002528/pjd/turbulence/plots/"

    srun python3 ../postprocessing/Plot_B_E.py "$input" "$output" --Lx 256.0 --Ly 256.0

    Required:
        --Lx        box length along x in code units (d_e)
        --Ly        box length along y in code units (d_e)
        --Lz        box length along z in code units (d_e); REQUIRED for a 3D run,
                    rejected for a 2D run

    Optional:
        --b0        mean guide field along z, i.e. setup.bguide. If non-zero it is
                    SUBTRACTED from the Bz panel so that dBz is plotted (default: 0)
        --t-eddy    outer-scale eddy turnover time in code units; copy the value
                    "Eddy turnover time [1/w_p1]" printed by the pgen. If given, the
                    title also reports t / t_eddy, the natural clock for decay
        --pct       percentile used for the colour limits (default: 99.5)
        --per-frame rescale colours independently for every snapshot. The DEFAULT is
                    the opposite: one fixed scale shared by all frames, which is what
                    you want for a movie of a DECAYING run (an amplitude that halves
                    must look halved, not get renormalised away)

    Bx and By share one colour scale, as do Ex and Ey, so that in-plane anisotropy is
    visible by eye. Bz, |B|, Ez and |E| each keep their own.

2D/3D compatibility
-------------------
    Entity writes field arrays with the SLOWEST axis first:
        2D run : (Ny, Nx)
        3D run : (Nz, Ny, Nx)        [axis0 = z, axis1 = y, axis2 = x]
    The dimensionality is taken from the presence of the X3 coordinate array, not from
    the shape of a field, and a 3D dump is cut at the mid-plane in z (axis 0).

    Variable names in the .bp files are unchanged (X1/X2/X3, fB1/fB2/fB3, fE1/fE2/fE3);
    only the axis and panel labels are written as x, y, z.
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
parser.add_argument("--b0",        type=float, default=0.0,   help="Mean field along z (setup.bguide); subtracted from the Bz panel")
parser.add_argument("--t-eddy",    type=float, default=None,  dest="t_eddy", help="Outer eddy turnover time [1/w_p1] from the pgen printout")
parser.add_argument("--pct",       type=float, default=99.5,  help="Percentile for colour limits")
parser.add_argument("--per-frame", action="store_true", dest="per_frame", help="Rescale colours per snapshot instead of using one fixed scale")

args      = parser.parse_args()
base      = args.base
outdir    = args.outdir
Lx        = args.Lx
Ly        = args.Ly
Lz        = args.Lz
b0        = args.b0
t_eddy    = args.t_eddy
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
BZ_LABEL = r"$\delta B_z$" if b0 != 0.0 else r"$B_z$"

PANELS = [
    (r"$B_x$",   "div", "seismic", "B_inplane"),
    (r"$B_y$",   "div", "seismic", "B_inplane"),
    (BZ_LABEL,   "div", "seismic", "B_z"),
    (r"$|B|$",   "seq", "inferno", "B_mag"),
    (r"$E_x$",   "div", "seismic", "E_inplane"),
    (r"$E_y$",   "div", "seismic", "E_inplane"),
    (r"$E_z$",   "div", "seismic", "E_z"),
    (r"$|E|$",   "seq", "inferno", "E_mag"),
]

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
    print(f"    Mean field B_z [B0] = {b0}" + ("  (subtracted from the Bz panel)" if b0 != 0.0 else ""), flush=True)
    if t_eddy is not None:
        print(f"    Eddy turnover  [1/w_p] = {t_eddy}", flush=True)
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

    #! |B| uses the FULL field including the guide field; only the Bz panel shows the
    #! fluctuation. Mixing the two in one figure is deliberate: |B| tells you about
    #! compressibility, dBz about the fluctuation structure.
    B_mag = np.sqrt(Bx**2 + By**2 + Bz**2) if have_B else None
    E_mag = np.sqrt(Ex**2 + Ey**2 + Ez**2) if have_E else None

    panels = {
        r"$B_x$":  Bx,
        r"$B_y$":  By,
        BZ_LABEL:  (Bz - b0) if Bz is not None else None,
        r"$|B|$":  B_mag,
        r"$E_x$":  Ex,
        r"$E_y$":  Ey,
        r"$E_z$":  Ez,
        r"$|E|$":  E_mag,
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
        time_label = f"$t\\,\\omega_{{p}} = {t_code:.1f}$   |   $t\\,c/L_x = {t_code / Lx:.3f}$"
        if t_eddy is not None:
            time_label += f"   |   $t/t_{{\\rm eddy}} = {t_code / t_eddy:.2f}$"
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

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

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