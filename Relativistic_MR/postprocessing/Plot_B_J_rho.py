"""
Created on Sat Apr 18 2026

@author: Pranab JD, Claude AI

Plot B_x, J_z and rho from the relativistic single-Harris-sheet Entity run.

Layout
------
    2D  (one plane, XY or YZ)          : 1 x 3   (B_x, J_z, rho)
    3D  (two orthogonal planes)        : 2 x 3
        row 1: XY plane (mid-Z);  row 2: YZ plane (mid-X)

Geometry (single Harris, reversal across Y): outflow = X, inflow = Y, guide = Z.

Usage
-----
    srun -n 8 python3 Plot_Bx_Jz_rho.py "$input" "$output" --Lx 500
    srun -n 8 python3 Plot_Bx_Jz_rho.py "$input" "$output" --Lx 750 --plane2d yz

    --Lx : box length (code units). Assumes Lx = Ly (2D) and Lx = Ly = Lz (3D).
           Every axis gets N_TICKS equidistant ticks: linspace(0, Lx, N_TICKS).

Array axis order (Entity)
-------------------------
    2D field arrays : (Ny, Nx)            [axis0=y, axis1=x]
    3D field arrays : (Nz, Ny, Nx)        [axis0=z, axis1=y, axis2=x]
      3D XY (mid-Z) : start=[k,0,0], count=[1,Ny,Nx]  -> (Ny, Nx)
      3D YZ (mid-X) : start=[0,0,i], count=[Nz,Ny,1]  -> (Nz,Ny) -> .T -> (Ny,Nz)
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

#! Colorbar limits. None -> automatic percentile. Set a number to fix the range.
BX_VMAX  = 3.0
JZ_VMAX  = 1.5
RHO_VMIN = 0.0
RHO_VMAX = 3.0

#! Percentiles used when the corresponding limit above is None.
BX_PCT  = 99.0
JZ_PCT  = 99.0
RHO_PCT = 99.0

N_TICKS = 5             #! equidistant ticks per axis (incl. 0 and Lx)

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
parser.add_argument("--Lx", type=float, required=True,
                    help="Box length in code units; assumes Lx = Ly (2D), Lx = Ly = Lz (3D)")
args     = parser.parse_args()
base     = args.base
outdir   = args.outdir
plane2d  = args.plane2d.lower()
Lx_cli   = args.Lx

if Lx_cli <= 0.0:
    parser.error("--Lx must be positive")

#! N_TICKS equidistant values 0..Lx; same array on every axis (Lx = Ly = Lz).
TICKS = np.linspace(0.0, Lx_cli, N_TICKS)

if rank == 0:
    os.makedirs(outdir, exist_ok=True)
comm.Barrier()

#! ============================================================
#! Find all files & distribute across ranks (round-robin)
#! ============================================================
files = sorted(glob.glob(f"{base}/fields.*.bp"))

if rank == 0:
    print(f"\n\nFound {len(files)} files", flush=True)
    print(f"    2D plane = {plane2d}", flush=True)
    print(f"    Lx       = {Lx_cli}   ticks = {TICKS}", flush=True)
    print(" ", flush=True)

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

#! ---- hyperslab slice readers (3D): read ONE plane, not the whole cube ----
def slab_xy(stream, name, k, Ny, Nx):
    #! mid-Z plane -> (Ny, Nx); reads 1*Ny*Nx elements, not Nz*Ny*Nx
    a = np.asarray(stream.read(name, start=[int(k), 0, 0], count=[1, int(Ny), int(Nx)]))
    return a.reshape(int(Ny), int(Nx))

def slab_yz(stream, name, i, Nz, Ny):
    #! mid-X plane -> (Ny, Nz); reads Nz*Ny*1 elements, then transpose
    a = np.asarray(stream.read(name, start=[0, 0, int(i)], count=[int(Nz), int(Ny), 1]))
    return a.reshape(int(Nz), int(Ny)).T

def sym_limits(data, fixed_vmax, pct):
    if fixed_vmax is not None:
        return -abs(fixed_vmax), abs(fixed_vmax)
    vmax = np.percentile(np.abs(data), pct)
    if vmax == 0.0:
        vmax = 1.0e-30
    return -vmax, vmax

def rho_limits(data, fixed_vmin, fixed_vmax, pct):
    vmin = fixed_vmin if fixed_vmin is not None else np.percentile(data, 100.0 - pct)
    vmax = fixed_vmax if fixed_vmax is not None else np.percentile(data, pct)
    if vmax <= vmin:
        vmax = vmin + 1.0e-30
    return vmin, vmax

def draw(ax, data, title, xlabel, ylabel, extent, cmap, vmin, vmax, fig):
    im = ax.imshow(data, origin="lower", aspect="equal", extent=extent, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xticks(TICKS)                          #! N_TICKS equidistant, code units
    ax.set_yticks(TICKS)
    ax.tick_params(axis="both", labelsize=10)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=12)

lbl_x = r"$x\ \omega_p/c$"
lbl_y = r"$y\ \omega_p/c$"
lbl_z = r"$z\ \omega_p/c$"

#! ============================================================
#! Loop over assigned files
#! ============================================================
for fname in files_local:

    step_idx = step_from_fname(fname)

    with Stream(fname, "r") as s:
        next(s.steps())

        #! coordinate arrays are 1D and tiny -> full read is fine, gives the dims
        x = np.asarray(s.read("X1")); Nx = x.size
        y = np.asarray(s.read("X2")); Ny = y.size
        try:
            z = np.asarray(s.read("X3")); Nz = z.size
        except Exception:
            z = None; Nz = 1

        t_code = read_time(s)
        is3d   = (z is not None and Nz > 1)

        if is3d:
            #! HYPERSLAB: read only the two plotted planes (mid-Z, mid-X)
            k = Nz // 2
            i = Nx // 2
            Bx_xy  = slab_xy(s, "fB1", k, Ny, Nx)
            Jz_xy  = slab_xy(s, "fJ3", k, Ny, Nx)
            rho_xy = slab_xy(s, "fN",  k, Ny, Nx)
            Bx_yz  = slab_yz(s, "fB1", i, Nz, Ny)
            Jz_yz  = slab_yz(s, "fJ3", i, Nz, Ny)
            rho_yz = slab_yz(s, "fN",  i, Nz, Ny)
        else:
            #! 2D planes are small -> full read
            Bx  = np.asarray(s.read("fB1"))
            Jz  = np.asarray(s.read("fJ3"))
            rho = np.asarray(s.read("fN"))

    #! ========================================================
    #! Time label: light-crossing times of Lx (t c/Lx = t_code/Lx, c=1)
    #! ========================================================
    Lx = float(x.max() - x.min())
    if (not np.isnan(t_code)) and Lx > 0:
        time_label = rf"$t\,c/L_x = {t_code / Lx:.2f}$"
    else:
        time_label = f"step {step_idx:09d}   (time or Lx not found)"

    #! ========================================================
    #! 2D : 1 x 3  (B_x, J_z, rho) in the chosen plane
    #! ========================================================
    if not is3d:
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
    #! 3D : 2 x 3.  Row1 = XY (mid-Z); Row2 = YZ (mid-X). Slabs already read.
    #! ========================================================
    else:
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