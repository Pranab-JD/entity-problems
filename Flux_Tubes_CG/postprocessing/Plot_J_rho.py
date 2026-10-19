"""
Created on Sat Apr 18 06:06 2026

@author: Pranab JD, Claude AI

Usage:

    input="/scratch/project_465002528/pjd/flux_tubes/tubes/fields/"
    output="/scratch/project_465002528/pjd/flux_tubes/tubes/plots/"

    srun python3 ../postprocessing/Plot_J_rho.py "$input" "$output" --Lx 256.0 --r_j 64.0

    Required:
        --Lx    box length along X in code units (sets the t c/L_x normalisation AND the x axis)
        --r_j   tube radius in code units        (sets the t c/R   normalisation)

    Optional:
        --Ly    box length along Y in code units (default: same as --Lx)

2D/3D compatibility
-------------------
    Fields from a 2D run have shape (Nx, Ny).
    Fields from a 3D run have shape (Nx, Ny, Nz).
    The script detects the dimensionality from the shape of the first field array after reading and
    slices accordingly:
      - 2D: use the array directly (no slicing needed).
      - 3D: take a slice at the mid-plane along the third axis (Nz // 2).

Force-free diagnostic
---------------------
    The sixth panel plots |J x B| / (|J| |B|) = sin(theta), the angle between J and B. This is the
    correct force-free metric: J x B = 0 (Lorentz force density zero) is the definition of force-free,
    so this quantity -> 0 in a perfect equilibrium and grows where J is NOT aligned with B.

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
parser.add_argument("--Lx",   type=float, required=True, help="Box length along X in code units; also sets the axis extent and ticks")
parser.add_argument("--Ly",   type=float, default=None,  help="Box length along Y in code units (default: same as --Lx)")
parser.add_argument("--r_j",  type=float, required=True, help="Tube radius in code length units")

args   = parser.parse_args()
base   = args.base
outdir = args.outdir
Lx     = args.Lx
Ly     = args.Ly if args.Ly is not None else args.Lx
r_j    = args.r_j

if Lx <= 0.0:
    parser.error("--Lx must be positive")
if Ly <= 0.0:
    parser.error("--Ly must be positive")
if r_j <= 0.0:
    parser.error("--r_j must be positive")

if rank == 0:
    os.makedirs(outdir, exist_ok=True)
comm.Barrier()

#! ============================================================
#! Helpers
#! ============================================================
files = sorted(glob.glob(f"{base}/fields.*.bp"))

if rank == 0:
    print(f"Found {len(files)} files", flush=True)
    print(f"    Box length L_x [d_e] = {Lx}", flush=True)
    print(f"    Box length L_y [d_e] = {Ly}", flush=True)
    print(f"    Tube radius R  [d_e] = {r_j}", flush=True)
    print(" ", flush=True)

files_local = files[rank::size]

def step_from_fname(fname):
    base_name = os.path.basename(fname)
    stem      = base_name.rsplit(".", 1)[0]
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

def set_equidistant_ticks(ax, xlo, xhi, ylo, yhi, n=N_TICKS):
    xt = np.linspace(xlo, xhi, n)
    yt = np.linspace(ylo, yhi, n)
    ax.set_xticks(xt)
    ax.set_yticks(yt)
    ax.set_xticklabels([f"{v:g}" for v in xt])
    ax.set_yticklabels([f"{v:g}" for v in yt])
    ax.minorticks_off()                     #! "no more, no less" -> no minor ticks either

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
    #! Physical time labels: BOTH are light crossing times (c = 1).
    #! ========================================================
    if not np.isnan(t_code):
        time_label = f"$t\\,c/L_x = {t_code / Lx:.3f}$   |   $t\\,c/R = {t_code / r_j:.3f}$"
    else:
        time_label = f"step {step_idx:09d}   (time not found — check TIME_KEY)"

    #! ========================================================
    #! Plot
    #! ========================================================
    fig, axs = plt.subplots(2, 3, figsize=(12, 6))

    field_limits = {
        "Jx":  0.3,
        "Jy":  0.3,
        "Jz":  0.6,
        "|J|": 0.6,
        "rho": 1.0,
        # Force-free metric: fix floor at 0 so the map reads as
        # "distance from force-free" rather than a stretched percentile band.
        "|JxB|/(|J||B|)": (0.0, 1.0),
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

    x_lo, x_hi = 0.0, Lx
    y_lo, y_hi = 0.0, Ly

    #! sanity check: the file should agree with --Lx/--Ly to within about one cell
    if rank == 0 and fname == files_local[0]:
        dx_est, dy_est = (x.max() - x.min()) / max(x.size - 1, 1), (y.max() - y.min()) / max(y.size - 1, 1)
        if abs(x.max() - x.min() + dx_est - Lx) > 2.0 * dx_est or abs(y.max() - y.min() + dy_est - Ly) > 2.0 * dy_est:
            print(f"  [warn] file extent is x:[{x.min():g},{x.max():g}] y:[{y.min():g},{y.max():g}] but --Lx={Lx:g} --Ly={Ly:g}", flush=True)

    for ax, (data, label) in zip(axs.flat, fields):

        limits = field_limits[label]
        cmap   = cmaps[label]

        if limits is None:
            #* automatic percentile scaling
            if cmap == "seismic":
                vmax = np.percentile(np.abs(data), 99)
                vmin = -vmax
            else:
                vmin = np.percentile(data, 1)
                vmax = np.percentile(data, 99)
        elif np.isscalar(limits):
            #* a single number means a symmetric range for diverging maps, and 0 -> limit for sequential ones
            if cmap == "seismic":
                vmin, vmax = -float(limits), float(limits)
            else:
                vmin, vmax = 0.0, float(limits)
        else:
            #* explicit (vmin, vmax) pair
            vmin, vmax = limits

        im = ax.imshow(data, origin="lower", aspect="equal", extent=[x_lo, x_hi, y_lo, y_hi], cmap=cmap, vmin=vmin, vmax=vmax)

        ax.set_title(label)
        ax.tick_params(axis="both", which="major", labelsize=10, length=6)
        ax.set_xlabel(r"$x [d_{e}]$")
        ax.set_ylabel(r"$y [d_{e}]$")

        #! exactly 5 equidistant ticks + labels on each axis of this panel
        set_equidistant_ticks(ax, x_lo, x_hi, y_lo, y_hi)

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(time_label, fontsize=16, y=1.00)
    fig.tight_layout()

    outfile = f"{outdir}/moments_{step_idx:09d}.png"
    fig.savefig(outfile, dpi=150)
    plt.close(fig)

    print(f"Saved {outfile}", flush=True)

comm.Barrier()

if rank == 0:
    print("\nDone.", flush=True)
    print("\n--------------------------\n\n", flush=True)