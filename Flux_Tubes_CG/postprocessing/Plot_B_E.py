"""
Created on Sat Apr 18 06:06 2026

@author: Pranab JD, Claude AI

Usage:

    input="/scratch/project_465002528/pjd/flux_tubes/tubes/fields/"
    output="/scratch/project_465002528/pjd/flux_tubes/tubes/plots/"

    srun python3 ../postprocessing/Plot_B_E.py "$input" "$output" --r_j 50.0

    Optional arguments:
        --r_j   tube radius in code length units (default: 50.0)

2D/3D compatibility
-------------------
    Fields from a 2D run have shape (Nx, Ny).
    Fields from a 3D run have shape (Nx, Ny, Nz).
    The script detects the dimensionality from the shape of the first
    field array after reading and slices accordingly:
      - 2D: use the array directly (no slicing needed).
      - 3D: take a slice at the mid-plane along the third axis (Nz // 2).

Time labels
-----------
    Filenames are fields.NNNNNNNNN.bp where NNNNNNNNN is the timestep index.
    Simulation time is read from the "time" variable/attribute in the BP file.
    ⚠ If Entity stores time under a different key, adjust TIME_KEY below.

    Two physical timescales are shown in the title:
      t / (R/c)    light crossing times of the tube radius r_j  (c = 1)
      t * omega_pe plasma frequency units; omega_pe = 1/skindepth = 1/1.0 = 1
                   so t_pe = t_code exactly (skindepth hardcoded to 1.0)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
from mpi4py import MPI
from adios2 import Stream
import argparse, os, glob
import matplotlib.pyplot as plt

#! ============================================================
#! Hardcoded physical constants
#! ============================================================
SKINDEPTH = 1.0   # d_e in code length units — hardcoded
                  # omega_pe = c / d_e = 1.0 / SKINDEPTH = 1.0
                  # so t * omega_pe = t_code * 1.0 = t_code exactly

TIME_KEY = "Time"

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
parser.add_argument("base",   type=str,
                    help="Directory with fields.NNNNNNNNN.bp files")
parser.add_argument("outdir", type=str,
                    help="Output directory for PNG plots")
parser.add_argument("--r_j",  type=float, default=50.0,
                    help="Tube radius in code length units (default: 50.0)")

args   = parser.parse_args()
base   = args.base
outdir = args.outdir
r_j    = args.r_j

if rank == 0:
    os.makedirs(outdir, exist_ok=True)
comm.Barrier()

#! ============================================================
#! Find all files & distribute across ranks
#! ============================================================
files = sorted(glob.glob(f"{base}/fields.*.bp"))

if rank == 0:
    print(f"Found {len(files)} files", flush=True)
    print(f"    skindepth (d_e) = {SKINDEPTH} (hardcoded)", flush=True)
    print(f"    r_j             = {r_j} [d_e]", flush=True)
    print(f"    omega_pe        = {1.0/SKINDEPTH:.4f}", flush=True)
    print(" ", flush=True)

files_local = files[rank::size]

#! ============================================================
#! Helper: extract timestep index from filename
#!   fields.000000001.bp -> 1
#!   fields.000000251.bp -> 251
#! ============================================================
def step_from_fname(fname):
    base_name = os.path.basename(fname)          # fields.000000251.bp
    stem      = base_name.rsplit(".", 1)[0]      # fields.000000251
    try:
        return int(stem.split(".")[-1])          # 251
    except ValueError:
        return -1

#! ============================================================
#! Helper: read simulation time from BP file.
#!   Tries variable first, then attribute.
#!   Returns NaN if neither is found — see ⚠ note on TIME_KEY.
#! ============================================================
def read_time(stream):
    # Matches the working shock postprocessing code:
    #   physical_time = s.read("Time")
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
#! Loop over assigned files
#! ============================================================
for fname in files_local:

    step_idx = step_from_fname(fname)

    with Stream(fname, "r") as s:
        next(s.steps())

        x  = np.asarray(s.read("X1"))
        y  = np.asarray(s.read("X2"))

        t_code = read_time(s)

        Bx = np.asarray(s.read("fB1"))
        By = np.asarray(s.read("fB2"))
        Bz = np.asarray(s.read("fB3"))

        Ex = np.asarray(s.read("fE1"))
        Ey = np.asarray(s.read("fE2"))
        Ez = np.asarray(s.read("fE3"))

    #! ========================================================
    #! Dimension detection and slicing
    #! ========================================================
    ndim = Bx.ndim
    if ndim == 3:
        z_half = Bx.shape[2] // 2
        Bx = Bx[:, :, z_half]; By = By[:, :, z_half]; Bz = Bz[:, :, z_half]
        Ex = Ex[:, :, z_half]; Ey = Ey[:, :, z_half]; Ez = Ez[:, :, z_half]
    elif ndim != 2:
        raise ValueError(f"Unexpected field array rank {ndim} in {fname}")

    B_mag = np.sqrt(Bx**2 + By**2 + Bz**2)
    E_mag = np.sqrt(Ex**2 + Ey**2 + Ez**2)

    #! ========================================================
    #! Physical time labels
    #!
    #!   t / (R/c)   = t_code / r_j
    #!                 light crossing times of the tube radius
    #!
    #!   t * omega_pe = t_code / SKINDEPTH = t_code / 1.0 = t_code
    #!                 plasma frequency units
    #!                 (since omega_pe = c/d_e = 1/1.0 = 1 in code units)
    #! ========================================================
    if not np.isnan(t_code):
        t_lc  = t_code / r_j
        t_pe  = t_code / SKINDEPTH          # = t_code since SKINDEPTH=1
        time_label = (f"$t\\,c/R = {t_lc:.3f}$   |   "
                      f"$t\\,\\omega_{{pe}} = {t_pe:.1f}$")
    else:
        time_label = f"step {step_idx:09d}   (time not found — check TIME_KEY)"

    #! ========================================================
    #! Plot
    #! ========================================================
    fig, axs = plt.subplots(2, 4, figsize=(16, 8))

    # Set to None for automatic percentile-based scaling,
    # or provide explicit (vmin, vmax) tuples to fix limits.
    field_limits = {
        "Bx":  None, "By":  None, "Bz":  None, "|B|": None,
        "Ex":  None, "Ey":  None, "Ez":  None, "|E|": None,
    }

    cmaps = {
        "Bx":  "seismic", "By":  "seismic",
        "Bz":  "inferno", "|B|": "inferno",
        "Ex":  "seismic", "Ey":  "seismic",
        "Ez":  "seismic", "|E|": "inferno",
    }

    fields = [
        (Bx, "Bx"), (By, "By"), (Bz, "Bz"), (B_mag, "|B|"),
        (Ex, "Ex"), (Ey, "Ey"), (Ez, "Ez"), (E_mag, "|E|"),
    ]

    for ax, (data, label) in zip(axs.flat, fields):

        limits = field_limits[label]
        cmap   = cmaps[label]

        if limits is not None:
            vmin, vmax = limits
        else:
            if cmap == "seismic":
                vmax = np.percentile(np.abs(data), 99)
                vmin = -vmax
            else:
                vmin = np.percentile(data, 1)
                vmax = np.percentile(data, 99)

        im = ax.imshow(
            data,
            origin="lower",
            aspect="equal",
            extent=[x.min(), x.max(), y.min(), y.max()],
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )

        ax.set_title(label)
        ax.tick_params(axis="both", which="major", labelsize=10, length=6)
        ax.tick_params(axis="both", which="minor", labelsize=8, length=3)
        ax.set_xlabel(r"$x\,/\,d_{e}$")
        ax.set_ylabel(r"$y\,/\,d_{e}$")

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(time_label, fontsize=16, y=1.00)
    fig.tight_layout()

    outfile = f"{outdir}/fields_{step_idx:09d}.png"
    fig.savefig(outfile, dpi=150)
    plt.close(fig)

    print(f"Saved {outfile}", flush=True)

#! ============================================================
#! Sync
#! ============================================================
comm.Barrier()

if rank == 0:
    print("\nDone.", flush=True)