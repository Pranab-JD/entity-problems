"""
Created on Sat Apr 18 06:06 2026

@author: Pranab JD, ChatGPT

Usage: 
    
    input="/scratch/project_465002528/pjd/flux_tubes/tubes/fields/"
    output="/scratch/project_465002528/pjd/flux_tubes/tubes/plots/"
    
    srun python3 ../postprocessing/Plot_B_E.py "$input" "$output"

"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
from mpi4py import MPI
from adios2 import Stream
import argparse, os, glob
import matplotlib.pyplot as plt

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
parser.add_argument("base", type=str, help="Directory with fields.*.bp")
parser.add_argument("outdir", type=str, help="Output directory")

args = parser.parse_args()
base = args.base
outdir = args.outdir

# only rank 0 creates output directory
if rank == 0:
    os.makedirs(outdir, exist_ok=True)
comm.Barrier()

#! ============================================================
#! Find all files & distribute across ranks
#! ============================================================
files = sorted(glob.glob(f"{base}/fields.*.bp"))

if rank == 0:
    print(f"Found {len(files)} files", flush=True)
    print(" ", flush=True)

files_local = files[rank::size]

#! ============================================================
#! Loop over all files
#! ============================================================
for fname in files_local:

    step_str = fname.split(".")[-2]

    with Stream(fname, "r") as s:
        next(s.steps())

        # coords
        x = np.asarray(s.read("X1"))
        y = np.asarray(s.read("X2"))
        # z = np.asarray(s.read("X3"))

        # Magnetic Field (3D)
        Bx = np.asarray(s.read("fB1"))
        By = np.asarray(s.read("fB2"))
        Bz = np.asarray(s.read("fB3"))

        # Electric Field (3D)
        Ex = np.asarray(s.read("fE1"))
        Ey = np.asarray(s.read("fE2"))
        Ez = np.asarray(s.read("fE3"))

        B_mag = np.sqrt(Bx**2 + By**2 + Bz**2)
        E_mag = np.sqrt(Ex**2 + Ey**2 + Ez**2)

    #! ========================================================
    #! Plot
    #! ========================================================
    fig, axs = plt.subplots(2, 4, figsize=(12, 6))

    field_limits = {"Bx": (-0.25, 0.25), "By": (-0.25, 0.25), "Bz": (0.0, 1.0), "|B|": (0.0, 1.5),
                    "Ex": (-0.25, 0.25), "Ey": (-0.25, 0.25), "Ez": (-0.25, 0.25), "|E|": (0.0, 0.4)}

    fields = [(Bx, "Bx"), (By, "By"), (Bz, "Bz"), (B_mag, "|B|"),
              (Ex, "Ex"), (Ey, "Ey"), (Ez, "Ez"), (E_mag, "|E|")]

    for ax, (data, title) in zip(axs.flat, fields):
        
        vmin, vmax = field_limits[title]

        im = ax.imshow(data, origin="lower", aspect="equal", extent=[x.min(), x.max(), y.min(), y.max()],
                             cmap="seismic", vmin=vmin, vmax=vmax)

        ax.set_title(title)
        ax.tick_params(axis="both", which="major", labelsize=10, length=6)
        ax.tick_params(axis="both", which="minor", labelsize=8, length=3)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    fig.tight_layout()

    outfile = f"{outdir}/fields_{step_str}.png"
    fig.savefig(outfile, dpi=150)
    plt.close(fig)

    print(f"Saved {outfile}", flush=True)

#! ============================================================
#! Sync
#! ============================================================

comm.Barrier()

if rank == 0:
    print("\nDone.",flush=True)