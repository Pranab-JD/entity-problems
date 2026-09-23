"""
Created on Sat Apr 18 2026

@author: Pranab JD, Claude AI

Plot B_x, J_z and rho from the relativistic single-Harris-sheet Entity run.

Layout
------
    2D  (one plane, XY or YZ)          : 1 x 3
        B_x        J_z        rho
    3D  (two orthogonal planes)        : 2 x 3
        B_x(X,Y)   J_z(X,Y)   rho(X,Y)      row 1: XY plane (mid-Z)
        B_x(Y,Z)   J_z(Y,Z)   rho(Y,Z)      row 2: YZ plane (mid-X)

Geometry (single Harris, reversal across Y): outflow = X, inflow = Y, guide = Z.

Usage
-----
    input="/scratch/project_465002528/pjd/RMR_ie_2D/fields/"
    output="/scratch/project_465002528/pjd/RMR_ie_2D/plots/"

    srun python3 Plot_Bx_Jz_rho.py "$input" "$output"            # 2D default: XY
    srun python3 Plot_Bx_Jz_rho.py "$input" "$output" --plane2d yz

Array axis order (Entity)
-------------------------
    2D field arrays : (Ny, Nx)            [axis0=y, axis1=x]
    3D field arrays : (Nz, Ny, Nx)        [axis0=z, axis1=y, axis2=x]
    imshow(M) maps row->vertical, col->horizontal.
      2D            : array as-is                 -> (Ny, Nx)
      3D XY (mid-Z) : A[Nz//2, :, :]              -> (Ny, Nx)
      3D YZ (mid-X) : A[:, :, Nx//2].T            -> (Nz,Ny) -> (Ny, Nz)

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

#! Colorbar limits. None -> automatic (symmetric percentile for diverging
#! B_x / J_z; 1-99 percentile for rho). Set a number to fix the range.
BX_VMAX  = 3.0          #! e.g. 0.3  -> B_x fixed to (-0.3, 0.3)
JZ_VMAX  = 0.5          #! e.g. 0.08 -> J_z fixed to (-0.08, 0.08)
RHO_VMIN = 0.5          #! e.g. 0.0  -> rho colour floor
RHO_VMAX = 3.0          #! e.g. 6.0  -> rho colour ceiling

#! Percentiles used when the corresponding limit above is None.
BX_PCT  = 99.0
JZ_PCT  = 99.0
RHO_PCT = 99.0

XTICKS = [0, 200, 400, 600, 800]          #! e.g. [0, 200, 400, 600, 800]
YTICKS = [0, 200, 400, 600, 800]          #! e.g. [0, 200, 400, 600, 800]

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
parser.add_argument("base",   type=str, help="Directory with fields.NNNNNNNNN.bp files")
parser.add_argument("outdir", type=str, help="Output directory for PNG plots")
parser.add_argument("--plane2d", type=str, default="xy", choices=["xy", "yz"],
                    help="For 2D runs: which plane the run is in (default: xy)")

args     = parser.parse_args()
base     = args.base
outdir   = args.outdir
plane2d  = args.plane2d.lower()

if rank == 0:
    os.makedirs(outdir, exist_ok=True)
comm.Barrier()

#! ============================================================
#! Find all files & distribute across ranks
#! ============================================================
files = sorted(glob.glob(f"{base}/fields.*.bp"))

if rank == 0:
    print(f"Found {len(files)} files", flush=True)
    print(f"    2D plane = {plane2d}", flush=True)
    print(" ", flush=True)

files_local = files[rank::size]

#! ============================================================
#! Helper: extract timestep index from filename
#!   fields.000000251.bp -> 251
#! ============================================================
def step_from_fname(fname):
    stem = os.path.basename(fname).rsplit(".", 1)[0]
    try:
        return int(stem.split(".")[-1])
    except ValueError:
        return -1

#! ============================================================
#! Helper: read simulation time (variable first, then attribute)
#! ============================================================
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

#! ============================================================
#! Colour-limit helpers
#! ============================================================
def sym_limits(data, fixed_vmax, pct):
    #! symmetric diverging limits for B_x / J_z
    if fixed_vmax is not None:
        return -abs(fixed_vmax), abs(fixed_vmax)
    vmax = np.percentile(np.abs(data), pct)
    if vmax == 0.0:
        vmax = 1.0e-30
    return -vmax, vmax

def rho_limits(data, fixed_vmin, fixed_vmax, pct):
    #! sequential limits for rho
    vmin = fixed_vmin if fixed_vmin is not None else np.percentile(data, 100.0 - pct)
    vmax = fixed_vmax if fixed_vmax is not None else np.percentile(data, pct)
    if vmax <= vmin:
        vmax = vmin + 1.0e-30
    return vmin, vmax

#! ============================================================
#! Helper: draw one panel
#! ============================================================
def draw(ax, data, title, xlabel, ylabel, extent, cmap, vmin, vmax, fig):
    
    im = ax.imshow(data, origin="lower", aspect="equal", extent=extent, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)

    #! explicit tick VALUES (code units); leave None for auto
    if XTICKS is not None:
        ax.set_xticks(XTICKS)
    if YTICKS is not None:
        ax.set_yticks(YTICKS)

    ax.tick_params(axis="both", labelsize=10)   #! label SIZE only
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=12)

#! ============================================================
#! Loop over assigned files
#! ============================================================
for fname in files_local:

    step_idx = step_from_fname(fname)

    with Stream(fname, "r") as s:
        next(s.steps())

        x = np.asarray(s.read("X1"))
        y = np.asarray(s.read("X2"))
        try:
            z = np.asarray(s.read("X3"))
        except Exception:
            z = None

        t_code = read_time(s)

        Bx  = np.asarray(s.read("fB1"))     #! reversing (reconnecting) field
        Jz  = np.asarray(s.read("fJ3"))     #! out-of-plane current
        rho = np.asarray(s.read("fN"))      #! number density

    ndim = Bx.ndim

    #! ========================================================
    #! Time label: light-crossing times of Lx  (t * c / Lx = t_code / Lx, c=1)
    #! ========================================================
    Lx = float(x.max() - x.min())      #! box length along X, code units
    if (not np.isnan(t_code)) and Lx > 0:
        t_lc = t_code / Lx
        time_label = rf"$t\,c/L_x = {t_lc:.2f}$"
    else:
        time_label = f"step {step_idx:09d}   (time or Lx not found)"

    lbl_x = r"$x\ \omega_p/c$"
    lbl_y = r"$y\ \omega_p/c$"
    lbl_z = r"$z\ \omega_p/c$"

    #! ========================================================
    #! 2D : 1 x 3  (B_x, J_z, rho) in the chosen plane
    #! ========================================================
    if ndim == 2:
        #! array is (N2, N1); horizontal axis is X (xy) or Z (yz), vertical is Y
        if plane2d == "xy":
            ext = [x.min(), x.max(), y.min(), y.max()]
            hlabel, hx = lbl_x, "X"
        else:  # yz
            ext = [z.min(), z.max(), y.min(), y.max()]
            hlabel, hx = lbl_z, "Z"

        fig, axs = plt.subplots(1, 3, figsize=(4.6 * 3, 4.7), constrained_layout=True)

        vmin, vmax = sym_limits(Bx, BX_VMAX, BX_PCT)
        draw(axs[0], Bx, rf"$B_x\ ({hx}, Y)$", hlabel, lbl_y, ext, "seismic", vmin, vmax, fig)

        vmin, vmax = sym_limits(Jz, JZ_VMAX, JZ_PCT)
        draw(axs[1], Jz, rf"$J_z\ ({hx}, Y)$", hlabel, lbl_y, ext, "seismic", vmin, vmax, fig)

        vmin, vmax = rho_limits(rho, RHO_VMIN, RHO_VMAX, RHO_PCT)
        draw(axs[2], rho, rf"$\rho\ ({hx}, Y)$", hlabel, lbl_y, ext, "inferno", vmin, vmax, fig)

    #! ========================================================
    #! 3D : 2 x 3.  Row1 = XY plane (mid-Z); Row2 = YZ plane (mid-X).
    #! ========================================================
    elif ndim == 3:
        Nz, Ny, Nx = Bx.shape
        k = Nz // 2
        i = Nx // 2

        #! XY plane (mid-Z): (Ny, Nx), row=Y col=X
        Bx_xy, Jz_xy, rho_xy = Bx[k, :, :], Jz[k, :, :], rho[k, :, :]
        #! YZ plane (mid-X): (Nz,Ny).T -> (Ny, Nz), row=Y col=Z
        Bx_yz, Jz_yz, rho_yz = Bx[:, :, i].T, Jz[:, :, i].T, rho[:, :, i].T

        ext_xy = [x.min(), x.max(), y.min(), y.max()]
        ext_yz = [z.min(), z.max(), y.min(), y.max()]

        fig, axs = plt.subplots(2, 3, figsize=(4.6 * 3, 5.2 * 2), constrained_layout=True)

        #! ---- row 1: XY ----
        vmin, vmax = sym_limits(Bx_xy, BX_VMAX, BX_PCT)
        draw(axs[0, 0], Bx_xy, r"$B_x\ (X, Y)$", lbl_x, lbl_y, ext_xy, "seismic", vmin, vmax, fig)
        vmin, vmax = sym_limits(Jz_xy, JZ_VMAX, JZ_PCT)
        draw(axs[0, 1], Jz_xy, r"$J_z\ (X, Y)$", lbl_x, lbl_y, ext_xy, "seismic", vmin, vmax, fig)
        vmin, vmax = rho_limits(rho_xy, RHO_VMIN, RHO_VMAX, RHO_PCT)
        draw(axs[0, 2], rho_xy, r"$\rho\ (X, Y)$", lbl_x, lbl_y, ext_xy, "inferno", vmin, vmax, fig)

        #! ---- row 2: YZ ----
        vmin, vmax = sym_limits(Bx_yz, BX_VMAX, BX_PCT)
        draw(axs[1, 0], Bx_yz, r"$B_x\ (Y, Z)$", lbl_z, lbl_y, ext_yz, "seismic", vmin, vmax, fig)
        vmin, vmax = sym_limits(Jz_yz, JZ_VMAX, JZ_PCT)
        draw(axs[1, 1], Jz_yz, r"$J_z\ (Y, Z)$", lbl_z, lbl_y, ext_yz, "seismic", vmin, vmax, fig)
        vmin, vmax = rho_limits(rho_yz, RHO_VMIN, RHO_VMAX, RHO_PCT)
        draw(axs[1, 2], rho_yz, r"$\rho\ (Y, Z)$", lbl_z, lbl_y, ext_yz, "inferno", vmin, vmax, fig)

    else:
        raise ValueError(f"Unexpected field rank {ndim} in {fname}")

    fig.suptitle(time_label, fontsize=18)

    outfile = f"{outdir}/B_J_rho_{step_idx:09d}.png"
    fig.savefig(outfile, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved {outfile}", flush=True)

#! ============================================================
#! Sync
#! ============================================================
comm.Barrier()

if rank == 0:
    print("\nDone.", flush=True)