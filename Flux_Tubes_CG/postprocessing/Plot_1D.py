"""
Plot 1D profiles along X at Y=Ly/2 and along Y at X=Lx/2.
Finds the array index closest to the physical midpoint.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
from mpi4py import MPI
from adios2 import Stream
import argparse
import os
import glob
import matplotlib.pyplot as plt

# ============================================================
SKINDEPTH = 1.0
TIME_KEY  = "Time"
# ============================================================

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

parser = argparse.ArgumentParser()
parser.add_argument("base",   type=str, help="Directory with fields.*.bp files")
parser.add_argument("outdir", type=str, help="Output directory for PNG plots")
parser.add_argument("--r_j",  type=float, default=50.0, help="Tube radius in d_e")
args = parser.parse_args()

if rank == 0:
    os.makedirs(args.outdir, exist_ok=True)
comm.Barrier()

files       = sorted(glob.glob(f"{args.base}/fields.*.bp"))
files_local = files[rank::size]

# ------------------------------------------------------------
def step_from_fname(fname):
    stem = os.path.basename(fname).rsplit(".", 1)[0]
    try:    return int(stem.split(".")[-1])
    except: return -1

def read_time(stream):
    for key in (TIME_KEY, "time", "t"):
        try:
            v = stream.read(key)
            if v is not None: return float(v)
        except: pass
        try:    return float(stream.read_attribute(key))
        except: pass
    return float("nan")

def slice_mid_z(arr):
    arr = np.asarray(arr)
    if arr.ndim == 2: return arr
    if arr.ndim == 3: return arr[:, :, arr.shape[2] // 2]
    raise ValueError(f"Unexpected rank {arr.ndim}")

def idx_nearest(coord_1d, target):
    """Return the index in coord_1d closest to target value."""
    return int(np.argmin(np.abs(np.asarray(coord_1d) - target)))

def make_1d_axis(coord_raw, n, axis):
    """Extract a 1D coordinate array from whatever shape adios2 returns."""
    c = np.asarray(coord_raw)
    if c.ndim == 1 and c.size == n: return c
    if c.ndim == 2:
        if axis == "x" and c.shape[0] == n: return c[:, 0]
        if axis == "y" and c.shape[1] == n: return c[0, :]
    return np.arange(n, dtype=float)

def time_label(t, r_j):
    if not np.isnan(t):
        return (f"$tc/R = {t/r_j:.3f}$   |   "
                f"$t\\omega_{{pe}} = {t/SKINDEPTH:.1f}$")
    return "time unknown"

def plot_profiles(axis_vals, profiles, xlabel, title, outfile):
    colors = {"B": "tab:blue", "E": "tab:orange", "J": "tab:green"}
    fig, axs = plt.subplots(3, 3, figsize=(14, 10), sharex=True)
    for ax, (label, vals) in zip(axs.flat, profiles):
        c = colors.get(label[0], "k")
        ax.plot(axis_vals, vals, lw=1.4, color=c)
        ax.axhline(0, lw=0.8, color="k", alpha=0.35)
        ax.set_title(label, fontsize=13)
        ax.grid(True, alpha=0.3)
    for ax in axs[-1]:
        ax.set_xlabel(xlabel)
    fig.suptitle(title, fontsize=14, y=0.998)
    fig.tight_layout()
    fig.savefig(outfile, dpi=150)
    plt.close(fig)
    print(f"Saved {outfile}", flush=True)

# ============================================================
for fname in files_local:
    step = step_from_fname(fname)

    with Stream(fname, "r") as s:
        next(s.steps())
        x_raw = np.asarray(s.read("X1"))
        y_raw = np.asarray(s.read("X2"))
        t     = read_time(s)
        Bx = slice_mid_z(s.read("fB1"))
        By = slice_mid_z(s.read("fB2"))
        Bz = slice_mid_z(s.read("fB3"))
        Ex = slice_mid_z(s.read("fE1"))
        Ey = slice_mid_z(s.read("fE2"))
        Ez = slice_mid_z(s.read("fE3"))
        Jx = slice_mid_z(s.read("fJ1"))
        Jy = slice_mid_z(s.read("fJ2"))
        Jz = slice_mid_z(s.read("fJ3"))

    # Entity field arrays have shape (Nx, Ny): first index = x, second = y
    nx, ny = Bx.shape

    x = make_1d_axis(x_raw, nx, "x")
    y = make_1d_axis(y_raw, ny, "y")

    # Middle array indices
    ix_mid = nx // 2
    iy_mid = ny // 2

    x_mid_val = x[ix_mid]
    y_mid_val = y[iy_mid]

    tl = time_label(t, args.r_j)

    fields = [("Bx",Bx),("By",By),("Bz",Bz),
              ("Ex",Ex),("Ey",Ey),("Ez",Ez),
              ("Jx",Jx),("Jy",Jy),("Jz",Jz)]

    # ---- Profile along Y at X middle index --------------------------
    # data has shape (Nx, Ny); data[ix_mid, :] gives values at fixed x,
    # varying over y — i.e. the y-profile at x middle index.
    profs_y = [(lbl, d[ix_mid, :]) for lbl, d in fields]
    title_y = (f"Along Y at X middle index: $x = {x_mid_val:.3f}$    |    {tl}")
    plot_profiles(y, profs_y, r"$y\,/\,d_e$", title_y,
                  f"{args.outdir}/Y_at_Xmid_{step:09d}.png")

    # ---- Profile along X at Y middle index --------------------------
    # data[:, iy_mid] gives values at fixed y, varying over x.
    profs_x = [(lbl, d[:, iy_mid]) for lbl, d in fields]
    title_x = (f"Along X at Y middle index: $y = {y_mid_val:.3f}$    |    {tl}")
    plot_profiles(x, profs_x, r"$x\,/\,d_e$", title_x,
                  f"{args.outdir}/X_at_Ymid_{step:09d}.png")

comm.Barrier()
if rank == 0:
    print("\nDone.", flush=True)