"""
Created on Sat Jun 13 17:44 2026

@author: Pranab JD, Claude AI

Plot kinetic, magnetic, and electric energy from Entity's *_stats.csv, all
normalised to the INITIAL TOTAL energy E_tot(t0) = KE + E_B + E_E.

Kinetic energy
--------------
    KE = T00 - Rho = <(gamma-1) m n>  (kinetic energy DENSITY, rest mass removed).
    T00 = <gamma m n> is the total particle energy density; Rho = <m n> is the
    rest-mass density. Both are built-in particle stats with the SAME
    normalisation (/(totVolume*ppc0)), so their difference is a clean,
    self-consistent kinetic energy density.

J.E (energy exchange channel)
-----------------------------
    The J.E column is the RATE of work the fields do on the particles (per volume):
        J.E > 0  -> fields give energy to particles (KE rises)
        J.E < 0  -> particles give energy to fields (KE falls)
    Its time-integral  int(J.E dt)  is the cumulative energy transferred to the
    particles.

Usage
-----
    folder="/scratch/project_465002528/pjd/flux_tubes_CG/2D_512_test/
    output="${folder}tubes/plots"

    srun python3 -u ../postprocessing/Energy_evolution.py "${folder}tubes_stats.csv" "$output" --ppc0 64

"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import argparse, os
import matplotlib.pyplot as plt

#! ============================================================
#! Args
#! ============================================================
parser = argparse.ArgumentParser()
parser.add_argument("csv",    type=str, help="Path to the *_stats.csv file")
parser.add_argument("outdir", type=str, help="Output directory for the PNG")
parser.add_argument("--ppc0", type=float, default=64.0,
                    help="particles.ppc0 from the TOML (default 64)")
args = parser.parse_args()

CSV    = args.csv
OUTDIR = args.outdir
PPC0   = args.ppc0
TCOL   = "time"   # hardcoded time column name
NREF   = 0        # hardcoded reference row for t0

os.makedirs(OUTDIR, exist_ok=True)

#! ============================================================
#! Read CSV (whitespace-padded, comma-separated, trailing comma per row)
#! ============================================================
with open(CSV, "r") as f:
    header_line = f.readline()
colnames = [c.strip() for c in header_line.strip().split(",")]
colnames = [c for c in colnames if c != ""]

data = np.genfromtxt(CSV, delimiter=",", skip_header=1)
if data.shape[1] == len(colnames) + 1 and np.all(np.isnan(data[:, -1])):
    data = data[:, :-1]

col = {name: data[:, i] for i, name in enumerate(colnames)}

def need(name):
    if name not in col:
        raise SystemExit(f"Column '{name}' not found. Available: {list(col.keys())}")
    return col[name]

t = need(TCOL)

#! ============================================================
#! Assemble energies (all /totVolume convention)
#! ============================================================
E_B   = 0.5 * (need("B1^2") + need("B2^2") + need("B3^2"))   # magnetic, total
E_Bip = 0.5 * (need("B1^2") + need("B2^2"))                  # magnetic, in-plane Bx^2+By^2
E_E   = 0.5 * (need("E1^2") + need("E2^2") + need("E3^2"))   # electric
KE    = need("T00") - need("Rho")                            # kinetic = T00 - Rho

E_tot      = KE + E_B + E_E
E_tot0     = E_tot[NREF]
if E_tot0 == 0 or not np.isfinite(E_tot0):
    raise SystemExit(f"Initial total energy at row {NREF} is {E_tot0}; cannot normalise.")

KE_f    = KE    / E_tot0
E_B_f   = E_B   / E_tot0
E_Bip_f = E_Bip / E_tot0     # in-plane magnetic; a SUBSET of E_B (not added to total)
E_E_f   = E_E   / E_tot0

#! ============================================================
#! Cumulative J.E:  int(J.E dt)  (energy transferred to particles)
#!   trapezoidal cumulative integral; normalise by E_tot0 to match the others.
#! ============================================================
JE = need("J.E")
# cumulative trapezoidal integral (no scipy/np.trapz dependency)
cumJE = np.zeros_like(JE)
for i in range(1, len(JE)):
    cumJE[i] = cumJE[i-1] + 0.5 * (JE[i] + JE[i-1]) * (t[i] - t[i-1])
cumJE_f = cumJE / E_tot0     # energy GIVEN TO particles by fields, in E_tot0 units

print()
print(f"Reference row {NREF}: t = {t[NREF]:.4g}, ppc0 = {PPC0}", flush=True)
print(f"  KE (t0)  = {KE[NREF]:.6g}   (= T00 - Rho)", flush=True)
print(f"  E_B(t0)  = {E_B[NREF]:.6g}", flush=True)
print(f"  E_E(t0)  = {E_E[NREF]:.6g}", flush=True)
print(f"  E_tot(t0)= {E_tot0:.6g}", flush=True)
print(f"  fractions:  KE={KE_f[NREF]:.4f}  B={E_B_f[NREF]:.4f}  E={E_E_f[NREF]:.4f}", flush=True)

#! ============================================================
#! Plot: energies + (-)cumulative J.E on log left axis;  raw J.E on right axis
#! ============================================================
fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)

ax.plot(t, KE_f,    color="red",   lw=1.8, label="Kinetic")
ax.plot(t, E_B_f,   color="blue",  lw=1.8, label="Magnetic")
ax.plot(t, E_Bip_f, color="cyan",  lw=1.8, label="Magnetic (excl. guide)")
ax.plot(t, E_E_f,   color="green", lw=1.8, label="Electric")

# cumulative energy transferred TO the fields = -int(J.E dt), normalised by E_tot0.
# (int(J.E dt) is energy to PARTICLES and is mostly negative; negating it gives a
#  positive "energy gained by fields" that fits the log energy axis.)
cum_to_fields_f = -cumJE / E_tot0
ax.plot(t, cum_to_fields_f, color="magenta", lw=1.8, ls="--",
        label=r"$-\int J\!\cdot\!E\,dt$ (to fields)")

ax.set_yscale("log")
ax.set_xlabel(r"$t\,\omega_{pe}$", fontsize=14)
ax.set_ylabel(r"$E(t) / E_{tot,0}$", fontsize=14)
ax.set_title("Energy evolution", fontsize=14)

# right axis: raw J.E (the instantaneous transfer rate), linear (changes sign)
ax2 = ax.twinx()
ax2.plot(t, JE, color="darkorange", lw=1.4, ls="--",
         label=r"$J\!\cdot\!E$ (rate)")
ax2.axhline(0.0, color="darkorange", lw=0.6, ls=":", alpha=0.4)
ax2.set_ylabel(r"$J\!\cdot\!E$", fontsize=14, color="darkorange")
ax2.tick_params(axis="y", labelcolor="darkorange")

# combined legend, placed OUTSIDE the plot box (to the right of ax2)
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, fontsize=11,
          loc="center left", bbox_to_anchor=(1.12, 0.5), frameon=True)

outfile = os.path.join(OUTDIR, "energy_history.png")
fig.savefig(outfile, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nSaved {outfile}", flush=True)