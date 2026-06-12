"""
Created on Sat Apr 18 06:06 2026

@author: Pranab JD, Claude AI

Usage:

    input="/scratch/project_465002528/pjd/flux_tubes/tubes/fields/"
    output="/scratch/project_465002528/pjd/flux_tubes/tubes/plots/"

    srun python3 ../postprocessing/Plot_J_rho.py "$input" "$output" --r_j 100.0

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
    Simulation time is read from the "Time" variable in the BP file.

    Two physical timescales are shown in the title:
      t / (R/c)    light crossing times of the tube radius r_j  (c = 1)
      t * omega_pe plasma frequency units; omega_pe = 1/skindepth = 1/1.0 = 1
                   so t_pe = t_code exactly (skindepth hardcoded to 1.0)

Force-free diagnostic
---------------------
    The sixth panel plots |J x B| / (|J| |B|) = sin(theta), the angle between
    J and B. This is the correct force-free metric: J x B = 0 (Lorentz force
    density zero) is the definition of force-free, so this quantity -> 0 in a
    perfect equilibrium and grows where J is NOT aligned with B.

    >>> CHECK: J and B must be co-located on the grid for this to be exact.
        If Entity stores J and B at different Yee positions, a half-cell
        stagger creates a spurious perpendicular component and makes the
        state look LESS force-free than it is. If the sin(theta) panel shows
        a one-cell-wide bright rim everywhere, suspect stagger, not physics.
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
                  # omega_pe = c / d_e = 1/1.0 = 1
                  # so t * omega_pe = t_code exactly

TIME_KEY  = "Time"

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
    base_name = os.path.basename(fname)
    stem      = base_name.rsplit(".", 1)[0]
    try:
        return int(stem.split(".")[-1])
    except ValueError:
        return -1

#! ============================================================
#! Helper: read simulation time from BP file.
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
#! Loop over assigned files
#! ============================================================
for fname in files_local:

    step_idx = step_from_fname(fname)

    with Stream(fname, "r") as s:
        next(s.steps())

        x = np.asarray(s.read("X1"))
        y = np.asarray(s.read("X2"))

        t_code = read_time(s)

        Jx  = np.asarray(s.read("fJ1"))
        Jy  = np.asarray(s.read("fJ2"))
        Jz  = np.asarray(s.read("fJ3"))
        rho = np.asarray(s.read("fN"))

        Bx = np.asarray(s.read("fB1"))
        By = np.asarray(s.read("fB2"))
        Bz = np.asarray(s.read("fB3"))

    #! ========================================================
    #! Dimension detection and slicing
    #! ========================================================
    ndim = Jx.ndim
    if ndim == 3:
        z_half = Jx.shape[2] // 2
        Jx  = Jx [:, :, z_half]
        Jy  = Jy [:, :, z_half]
        Jz  = Jz [:, :, z_half]
        rho = rho[:, :, z_half]
        Bx  = Bx [:, :, z_half]
        By  = By [:, :, z_half]
        Bz  = Bz [:, :, z_half]
    elif ndim != 2:
        raise ValueError(f"Unexpected field array rank {ndim} in {fname}")

    J_mag = np.sqrt(Jx**2 + Jy**2 + Jz**2)
    B_mag = np.sqrt(Bx**2 + By**2 + Bz**2)

    #! ========================================================
    #! Force-free metric: |J x B| / (|J| |B|) = sin(theta)
    #!   (J x B)_x = Jy*Bz - Jz*By
    #!   (J x B)_y = Jz*Bx - Jx*Bz
    #!   (J x B)_z = Jx*By - Jy*Bx
    #! Force-free  <=>  J x B = 0  <=>  this -> 0.
    #! ========================================================
    JxB_x = Jy * Bz - Jz * By
    JxB_y = Jz * Bx - Jx * Bz
    JxB_z = Jx * By - Jy * Bx
    JxB_mag = np.sqrt(JxB_x**2 + JxB_y**2 + JxB_z**2)

    denom = J_mag * B_mag
    JxB_norm = np.zeros_like(JxB_mag)
    mask = denom > 0.0
    JxB_norm[mask] = JxB_mag[mask] / denom[mask]
    # JxB_norm is sin(theta) in [0, 1]; 0 = force-free.

    #! ========================================================
    #! Physical time labels
    #! ========================================================
    if not np.isnan(t_code):
        t_lc = t_code / r_j
        t_pe = t_code / SKINDEPTH
        time_label = (f"$t\\,c/R = {t_lc:.3f}$   |   "
                      f"$t\\,\\omega_{{pe}} = {t_pe:.1f}$")
    else:
        time_label = f"step {step_idx:09d}   (time not found — check TIME_KEY)"

    #! ========================================================
    #! Plot
    #! ========================================================
    fig, axs = plt.subplots(2, 3, figsize=(12, 6))

    field_limits = {
        "Jx":  None,
        "Jy":  None,
        "Jz":  None,
        "|J|": None,
        "rho": None,
        # Force-free metric: fix floor at 0 so the map reads as
        # "distance from force-free" rather than a stretched percentile band.
        "|JxB|/(|J||B|)": (0.0, None),
    }

    cmaps = {
        "Jx":  "seismic",
        "Jy":  "seismic",
        "Jz":  "seismic",
        "|J|": "inferno",
        "rho": "inferno",
        "|JxB|/(|J||B|)": "inferno",
    }

    fields = [
        (Jx,       "Jx"),
        (Jy,       "Jy"),
        (Jz,       "Jz"),
        (J_mag,    "|J|"),
        (rho,      "rho"),
        (JxB_norm, "|JxB|/(|J||B|)"),
    ]

    for ax, (data, label) in zip(axs.flat, fields):

        limits = field_limits[label]
        cmap   = cmaps[label]

        if limits is not None:
            vmin, vmax = limits
            # Allow a fixed floor with auto top (vmax = None -> 99th pct).
            if vmax is None:
                vmax = np.percentile(data, 99)
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

    outfile = f"{outdir}/moments_{step_idx:09d}.png"
    fig.savefig(outfile, dpi=150)
    plt.close(fig)

    print(f"Saved {outfile}", flush=True)

#! ============================================================
#! Sync
#! ============================================================
comm.Barrier()

if rank == 0:
    print("\nDone.", flush=True)