"""
Created on Sat Sep 28 2026

@author: Pranab JD, Claude AI

Plot B_x, J_z and rho from the relativistic single-Harris-sheet Entity run.
Geometry (reversal across Y): outflow = X, inflow = Y, guide = Z.

Layout
------
    2D (one plane, XY or YZ)    : 1 x 3   B_x, J_z, rho
    3D (two orthogonal planes)  : 2 x 3   row 1 = XY at mid-Z, row 2 = ZY at mid-X

Array axis order (Entity)
-------------------------
    2D field arrays : (Ny, Nx)       [axis0 = x2, axis1 = x1]
    3D field arrays : (Nz, Ny, Nx)   [axis0 = x3, axis1 = x2, axis2 = x1]

Time is always normalised by the light-crossing time of the outflow direction, t c / L_x.

Usage
-----
    sfolder="/scratch/project_465003132/RMR_pair_3D/sigma_10_Z100/"
    folder="/scratch/project_465003132/RMR_pair_3D/sigma_10_Z100/RMR/"
    fields="${folder}/fields"
    output="${folder}/plots"

    Lx=200.0; Ly=100.0; Lz=100.0

    srun python3 -u ../postprocessing/Plot_B_J_rho.py "$fields" "$output" \
        --Lx "$Lx" --Ly "$Ly" --Lz "$Lz"        # 3D runs

    srun python3 -u ../postprocessing/Plot_B_J_rho.py "$fields" "$output" \
        --Lx "$Lx" --Ly "$Ly"                   # 2D runs in the XY plane

    srun python3 -u ../postprocessing/Plot_B_J_rho.py "$fields" "$output" \
        --Ly "$Ly" --Lz "$Lz" --plane2d yz      # 2D runs in the YZ plane
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
TIME_KEY = "Time"        #! adjust if Entity stores time under another key
F_BX     = "fB1"         #! in-plane reversing field
F_JZ     = "fJ3"         #! out-of-plane current
F_N      = "fN"          #! number density

#! Plot aspect: "equal" preserves the physical X/Y scale; "auto" fills the subplot
PLOT_ASPECT = "equal"    #! "equal" or "auto"

#! Colourbar limits. None -> percentile from the data.
BX_VMAX  = 2.0
JZ_VMAX  = 2.0
RHO_VMIN = 0.0
RHO_VMAX = 8.0
PCT      = 99.0               #! percentile used wherever a limit above is None

#! One panel's plot area in inches. PAD_W must cover the ylabel, the colourbar and its tick labels;
#! PAD_H covers the title and the xlabel.
PANEL, PAD_W, PAD_H = 4.6, 1.8, 1.2

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
parser.add_argument("base",      type=str)
parser.add_argument("outdir",    type=str)
parser.add_argument("--plane2d", type=str, default="xy", choices=["xy", "yz"],
                    help="2D runs only: which physical plane the run is in. Labels only; "
                         "the array is always (x2, x1). Default: xy")

#! Optional span overrides. None -> take the span from the file's coordinate arrays.
parser.add_argument("--Lx", type=float, default=None)
parser.add_argument("--Ly", type=float, default=None)
parser.add_argument("--Lz", type=float, default=None)
args = parser.parse_args()

BASE, OUTDIR, PLANE2D = args.base, args.outdir, args.plane2d.lower()
L_OVER = {"x1": args.Lx, "x2": args.Ly, "x3": args.Lz}      #! None = use the file
L_FLAG = {"x1": "--Lx", "x2": "--Ly", "x3": "--Lz"}         #! for the mismatch message

if PLOT_ASPECT not in ("equal", "auto"):
    parser.error('PLOT_ASPECT must be either "equal" or "auto"')

for k, v in L_OVER.items():
    if v is not None and v <= 0.0:
        parser.error(f"{L_FLAG[k]} override must be positive")

if rank == 0:
    os.makedirs(OUTDIR, exist_ok=True)
comm.Barrier()

files = sorted(glob.glob(f"{BASE}/fields.*.bp"))
if rank == 0:
    print(f"\nFound {len(files)} files   (2D plane = {PLANE2D})", flush=True)
    print(f"    L override = {{x1: {args.Lx}, x2: {args.Ly}, x3: {args.Lz}}}   (None -> read from the file)\n", flush=True)
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
    for getter in (lambda: stream.read(TIME_KEY), lambda: stream.read_attribute(TIME_KEY)):
        try:
            v = getter()
            if v is not None:
                return float(np.asarray(v).ravel()[0])
        except Exception:
            pass
    return float("nan")

_warned = set()

def warn_once(key, msg):
    """One line per distinct issue, from rank 0 only, to keep the logs readable."""
    if rank == 0 and key not in _warned:
        _warned.add(key)
        print(f"WARNING: {msg}", flush=True)

def axis_span(coord, name):
    """
    Span of one axis from its cell CENTRES: X1/X2/X3 have the same length as the field axis, so the
    domain runs from c[0] - dx/2 to c[-1] + dx/2. The plotted axis is pinned to [0, L]. Entity writes
    ascending, uniformly spaced coordinates; both are flagged if they do not hold, since a descending
    axis would give a negative span and an inverted extent. A --Lx/--Ly/--Lz override wins over the
    file, and is flagged if the two disagree by more than 1%.
    """
    c = np.asarray(coord, dtype=float).ravel()
    if c.size < 2:
        return 1.0
    d  = np.diff(c)
    dc = float(np.median(d))
    if dc <= 0.0:
        warn_once(f"desc_{name}", f"{name} is not increasing; the extent will be inverted")
    if np.max(np.abs(d - dc)) > 1.0e-6 * abs(dc):
        warn_once(f"nonuni_{name}", f"{name} spacing is non-uniform; the span uses the median dx")
    span  = float((c[-1] + 0.5 * dc) - (c[0] - 0.5 * dc))
    L_cli = L_OVER.get(name)
    if L_cli is None:
        return span
    if span > 0.0 and abs(span - L_cli) / span > 0.01:
        warn_once(f"mismatch_{name}",
                  f"{L_FLAG[name]} = {L_cli:g} disagrees with the {name} span {span:g} from the file "
                  f"by more than 1%; using the override")
    return float(L_cli)

def slab_xy(stream, name, k, Ny, Nx):
    """Mid-Z plane of a 3D array -> (Ny, Nx). Reads 1*Ny*Nx elements, not Nz*Ny*Nx."""
    a = np.asarray(stream.read(name, start=[int(k), 0, 0], count=[1, int(Ny), int(Nx)]))
    return a.reshape(int(Ny), int(Nx))

def slab_zy(stream, name, i, Nz, Ny):
    """Mid-X plane of a 3D array -> (Ny, Nz): rows = Y (vertical), columns = Z (horizontal)."""
    a = np.asarray(stream.read(name, start=[0, 0, int(i)], count=[int(Nz), int(Ny), 1]))
    return a.reshape(int(Nz), int(Ny)).T

def sym_limits(data, fixed):
    """Symmetric limits about zero, for the signed quantities."""
    v = abs(fixed) if fixed is not None else max(np.percentile(np.abs(data), PCT), 1.0e-30)
    return -v, v

def rho_limits(data):
    vmin = RHO_VMIN if RHO_VMIN is not None else np.percentile(data, 100.0 - PCT)
    vmax = RHO_VMAX if RHO_VMAX is not None else np.percentile(data, PCT)
    return vmin, max(vmax, vmin + 1.0e-30)

def draw(fig, ax, data, title, xlabel, ylabel, extent, cmap, vmin, vmax):
    im = ax.imshow(data, origin="lower", aspect=PLOT_ASPECT, extent=extent, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.tick_params(axis="both", labelsize=10)
    #! the colourbar is attached to the axes, never an inset: constrained_layout then reserves room for
    #! it AND for the next panel's tick labels, so nothing can overlap whatever the panel aspect is
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.ax.tick_params(labelsize=11)

def figure_size(row_aspects):
    """
    Figure size in inches for a 3-column grid, one entry per row giving that row's physical Ly/Lx.
    With PLOT_ASPECT = "equal" a wide box is short, so each row's height follows its own data aspect
    (clamped so a very flat or very tall box stays legible). With "auto" every panel is PANEL tall.
    """
    width = 3 * (PANEL + PAD_W)
    if PLOT_ASPECT == "equal":
        height = sum(min(max(PANEL * a, 1.5), 2.0 * PANEL) + PAD_H for a in row_aspects)
    else:
        height = len(row_aspects) * (PANEL + PAD_H)
    return width, height

LBL_X, LBL_Y, LBL_Z = r"$x\ \omega_p/c$", r"$y\ \omega_p/c$", r"$z\ \omega_p/c$"

#! ============================================================
#! Loop over the files assigned to this rank
#! ============================================================
for fname in files_local:
    step_idx = step_from_fname(fname)

    with Stream(fname, "r") as s:
        next(s.steps())
        #! coordinate arrays are 1D and tiny, so a full read is fine and gives the dimensions
        x = np.asarray(s.read("X1")); Nx = x.size
        y = np.asarray(s.read("X2")); Ny = y.size
        try:
            z = np.asarray(s.read("X3")); Nz = z.size
        except Exception:
            z, Nz = None, 1
        is3d   = (z is not None and Nz > 1)
        t_code = read_time(s)

        if is3d:
            k, i = Nz // 2, Nx // 2      #! the two plotted planes: mid-Z and mid-X
            Bx_xy, Jz_xy, rho_xy = (slab_xy(s, n, k, Ny, Nx) for n in (F_BX, F_JZ, F_N))
            Bx_zy, Jz_zy, rho_zy = (slab_zy(s, n, i, Nz, Ny) for n in (F_BX, F_JZ, F_N))
        else:
            Bx, Jz, rho = (np.asarray(s.read(n)) for n in (F_BX, F_JZ, F_N))

    #! ---- per-axis spans; no assumption that Lx = Ly = Lz ----
    Lx, Ly = axis_span(x, "x1"), axis_span(y, "x2")
    Lz     = axis_span(z, "x3") if is3d else None

    #! ---- time in light-crossing times of the outflow direction ----
    time_label = (rf"$t\,c/L_x = {t_code / Lx:.2f}$" if np.isfinite(t_code) and Lx > 0
                  else f"step {step_idx:09d}")

    if not is3d:
        #! the array is (Nx2, Nx1): horizontal = x1, vertical = x2 = Y. plane2d only sets the label.
        ext    = [0.0, Lx, 0.0, Ly]
        hlabel = LBL_X if PLANE2D == "xy" else LBL_Z
        hx     = "X"   if PLANE2D == "xy" else "Z"

        fig, axs = plt.subplots(1, 3, figsize=figure_size([Ly / Lx]), constrained_layout=True)
        draw(fig, axs[0], Bx,  rf"$B_x\ ({hx}, Y)$",  hlabel, LBL_Y, ext, "seismic", *sym_limits(Bx, BX_VMAX))
        draw(fig, axs[1], Jz,  rf"$J_z\ ({hx}, Y)$",  hlabel, LBL_Y, ext, "seismic", *sym_limits(Jz, JZ_VMAX))
        draw(fig, axs[2], rho, rf"$\rho\ ({hx}, Y)$", hlabel, LBL_Y, ext, "inferno", *rho_limits(rho))

    else:
        #! row 1 = XY at mid-Z (horizontal span Lx), row 2 = ZY at mid-X (horizontal span Lz);
        #! both rows share the vertical Y axis
        ext_xy, ext_zy = [0.0, Lx, 0.0, Ly], [0.0, Lz, 0.0, Ly]

        fig, axs = plt.subplots(2, 3, figsize=figure_size([Ly / Lx, Ly / Lz]), constrained_layout=True)
        draw(fig, axs[0, 0], Bx_xy,  r"$B_x\ (X, Y)$",  LBL_X, LBL_Y, ext_xy, "seismic", *sym_limits(Bx_xy, BX_VMAX))
        draw(fig, axs[0, 1], Jz_xy,  r"$J_z\ (X, Y)$",  LBL_X, LBL_Y, ext_xy, "seismic", *sym_limits(Jz_xy, JZ_VMAX))
        draw(fig, axs[0, 2], rho_xy, r"$\rho\ (X, Y)$", LBL_X, LBL_Y, ext_xy, "inferno", *rho_limits(rho_xy))
        draw(fig, axs[1, 0], Bx_zy,  r"$B_x\ (Z, Y)$",  LBL_Z, LBL_Y, ext_zy, "seismic", *sym_limits(Bx_zy, BX_VMAX))
        draw(fig, axs[1, 1], Jz_zy,  r"$J_z\ (Z, Y)$",  LBL_Z, LBL_Y, ext_zy, "seismic", *sym_limits(Jz_zy, JZ_VMAX))
        draw(fig, axs[1, 2], rho_zy, r"$\rho\ (Z, Y)$", LBL_Z, LBL_Y, ext_zy, "inferno", *rho_limits(rho_zy))

    fig.suptitle(time_label, fontsize=18)
    outfile = f"{OUTDIR}/B_J_rho_{step_idx:09d}.png"
    fig.savefig(outfile, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {outfile}", flush=True)

comm.Barrier()
if rank == 0:
    print("\nDone.", flush=True)