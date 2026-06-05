"""
Created on Sat Apr 18 06:06 2026

@author: Pranab JD, ChatGPT

Usage: 
    
    input="/scratch/project_465002528/pjd/flux_tubes/tubes/fields/"
    output="/scratch/project_465002528/pjd/flux_tubes/tubes/plots/"
    
    srun python3 ../postprocessing/Plot_J_rho.py "$input" "$output"

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

        # Current (3D)
        Jx = np.asarray(s.read("fJ1"))
        Jy = np.asarray(s.read("fJ2"))
        Jz = np.asarray(s.read("fJ3"))

        # Density (3D)
        rho = np.asarray(s.read("fN"))

        J_mag = np.sqrt(Jx**2 + Jy**2 + Jz**2)

        #! ========================================================
        #! Plot
        #! ========================================================
        fig, axs = plt.subplots(2, 3, figsize=(16, 6))

        field_limits = {
            "Jx":  None,
            "Jy":  None,
            "Jz":  None,
            "|J|": None,
            "rho": None,
        }

        cmaps = {
            "Jx":  "seismic",
            "Jy":  "seismic",
            "Jz":  "seismic",
            "|J|": "inferno",
            "rho": "inferno",
        }

        fields = [(Jx, "Jx"), (Jy, "Jy"), (Jz, "Jz"), (J_mag, "|J|"), (rho, "rho")]

        for ax, (data, title) in zip(axs.flat, fields):

            limits = field_limits[title]
            cmap   = cmaps[title]

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

            ax.set_title(title)
            ax.tick_params(axis="both", which="major", labelsize=10, length=6)
            ax.tick_params(axis="both", which="minor", labelsize=8, length=3)
            ax.set_xlabel("X")
            ax.set_ylabel("Y")

            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        fig.tight_layout()

        outfile = f"{outdir}/moments_{step_str}.png"
        fig.savefig(outfile, dpi=150)
        plt.close(fig)

        print(f"Saved {outfile}", flush=True)

#! ============================================================
#! Sync
#! ============================================================

comm.Barrier()

if rank == 0:
    print("\nDone.",flush=True)