"""
Created on Tue Oct 06 2026

@author: Pranab JD, Claude AI

Plot J_z, E_parallel, E.J and B_x in the XZ plane, on two Y-slices straddling the current sheet,
for every available output of a relativistic single-Harris-sheet Entity run.
Geometry (reversal across Y): outflow = X, inflow = Y, guide = Z.

Layout (one figure per output file, i.e. per time)
-------------------------------------------------
    2 x 4     row 1 = XZ plane at y = Ly/2 + YOFF cells   J_z | |E_par| | E.J | B_x
              row 2 = XZ plane at y = Ly/2 - YOFF cells   J_z | |E_par| | E.J | B_x

    Horizontal axis = X (outflow), vertical axis = Z (guide). 3D runs only: an XZ plane does not
    exist in a 2D (x1, x2) run, so the script stops with a message on those.

Array axis order (Entity)
-------------------------
    3D field arrays : (Nz, Ny, Nx)   [axis0 = x3, axis1 = x2, axis2 = x1]
    XZ slice at fixed y index j : start = [0, j, 0], count = [Nz, 1, Nx]  ->  (Nz, Nx)
    which is already rows = Z (vertical), columns = X (horizontal). No transpose.

E_parallel  (--epar)
--------------------
    "abs"    |E_par| = |(E.B)/|B||   [B0]     DEFAULT, sequential scale from 0
    "par"     E_par  =  (E.B)/|B|    [B0]     signed, symmetric scale
    "edotb"   E.B                    [B0^2]   signed, symmetric scale, Lorentz invariant

    WHY |E_par| IS THE DEFAULT. E_par flips sign under B -> -B, and the slicing plane sits near the
    neutral surface, which wanders in y. Crossing from one side of that surface to the other
    therefore reverses the sign for purely geometric reasons, so a signed map shows mostly WHICH
    SIDE of the sheet each cell is on rather than the non-ideal field. The modulus removes that
    artefact. Use "par" only when the sign itself is the question and the plane is known to sit
    cleanly on one side.

    "abs" and "par" need a guard where |B| -> 0: cells with |B| below EPS_FRAC * median(|B|) of that
    slice are blanked (NaN), because the parallel direction is undefined there and the ratio blows
    up. "edotb" has no such pathology and is the safer choice if the blanked fraction is large. Note
    (E.B)/|B| and E.B are NOT the same map rescaled: dividing by the local |B| is a spatially varying
    weight, so they differ in structure, not just in units.

E.J
---
    E.J = E1 J1 + E2 J2 + E3 J3, in units of B0 n0 q0 c, signed and symmetric about zero.
    This is the TOTAL power transfer from fields to plasma and includes the ideal (convective)
    contribution, so it marks outflows as strongly as X-lines. It corroborates E_par; it does not
    localise dissipation on its own. The sign is the content (positive = field -> plasma), so this
    one is never plotted as a modulus.

B_x  (fourth panel)
-------------------
    The reversing in-plane field, in units of B0, signed and symmetric about zero. On an XZ cut its
    ZERO CONTOUR is exactly where the slicing plane crosses the neutral surface, so this panel is
    the diagnostic for whether the two Y-slices straddle the sheet at all: a plane that sits cleanly
    above the sheet is uniformly one colour, while a mottled panel means the cut is weaving in and
    out of the layer and the other three panels are being sampled on both sides of the reversal.

Time is always normalised by the light-crossing time of the outflow direction, t c / L_x.

Usage
-----
    sfolder="/scratch/project_465003132/RMR_pair_3D/sigma_10_Z100/"
    folder="/scratch/project_465003132/RMR_pair_3D/sigma_10_Z100/RMR/"
    fields="${folder}/fields"
    output="${folder}/plots_Epar"

    Lx=200.0; Ly=100.0; Lz=100.0

    srun python3 -u ../postprocessing/Plot_Jz_Epar.py "$fields" "$output" \
        --Lx "$Lx" --Ly "$Ly" --Lz "$Lz"                     # y = Ny/2 +/- 1 cell

    srun python3 -u ../postprocessing/Plot_Jz_Epar.py "$fields" "$output" \
        --Lx "$Lx" --Ly "$Ly" --Lz "$Lz" --yoff 4            # y = Ny/2 +/- 4 cells

    srun python3 -u ../postprocessing/Plot_Jz_Epar.py "$fields" "$output" \
        --Lx "$Lx" --Ly "$Ly" --Lz "$Lz" --yidx 260 250      # explicit cell indices (top, bottom)

    srun python3 -u ../postprocessing/Plot_Jz_Epar.py "$fields" "$output" \
        --Lx "$Lx" --Ly "$Ly" --Lz "$Lz" --epar par          # SIGNED E_par instead of |E_par|

    srun python3 -u ../postprocessing/Plot_Jz_Epar.py "$fields" "$output" \
        --Lx "$Lx" --Ly "$Ly" --Lz "$Lz" --epar edotb        # E.B instead of (E.B)/|B|
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
F_E      = ("fE1", "fE2", "fE3")      #! E components, in that order
F_B      = ("fB1", "fB2", "fB3")      #! B components, in that order
F_J      = ("fJ1", "fJ2", "fJ3")      #! J components, in that order
F_JZ     = F_J[2]                     #! out-of-plane current, plotted on its own in column 1

#! ASPECT_XZ applies to both XZ panels. "equal", "auto", or a positive number (height/width).
ASPECT_XZ = 0.5

#! Colourbar limits. None -> percentile from the data (recommended for E_par and E.J, whose
#! magnitudes change by orders of magnitude between t = 0 and the saturated state).
JZ_VMAX   = 1.0
EPAR_VMAX = 0.05
EJ_VMAX   = 0.4
BX_VMAX   = 1.0
PCT       = 99.0              #! percentile used wherever a limit above is None

#! Colour maps. The signed quantities get a diverging map centred on zero; |E_par| is non-negative
#! and gets a sequential one, so "no parallel field" reads as the dark end rather than as mid-grey.
CMAP_SIGNED, CMAP_ABS = "seismic", "inferno"

#! |B| guard for EPAR_MODE "abs"/"par": blank cells with |B| < EPS_FRAC * median(|B|) of that slice
EPS_FRAC  = 0.05

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
parser.add_argument("base",   type=str)
parser.add_argument("outdir", type=str)
parser.add_argument("--yoff", type=int, default=1,
                    help="Offset in CELLS from the sheet centre for the two XZ slices. "
                         "Row 1 = Ny//2 + yoff, row 2 = Ny//2 - yoff. Default: 1")
parser.add_argument("--yidx", type=int, nargs=2, default=None, metavar=("JTOP", "JBOT"),
                    help="Explicit cell indices for the two slices; overrides --yoff")
parser.add_argument("--epar", type=str, default="abs", choices=["abs", "par", "edotb"],
                    help='"abs" = |(E.B)/|B|| in B0 (default, sign-flip artefact removed); '
                         '"par" = signed (E.B)/|B|; "edotb" = signed E.B in B0^2, invariant')

#! Optional span overrides. None -> take the span from the file's coordinate arrays.
parser.add_argument("--Lx", type=float, default=None)
parser.add_argument("--Ly", type=float, default=None)
parser.add_argument("--Lz", type=float, default=None)
args = parser.parse_args()

BASE, OUTDIR = args.base, args.outdir
YOFF, YIDX, EPAR_MODE = args.yoff, args.yidx, args.epar
L_OVER = {"x1": args.Lx, "x2": args.Ly, "x3": args.Lz}      #! None = use the file
L_FLAG = {"x1": "--Lx", "x2": "--Ly", "x3": "--Lz"}         #! for the mismatch message

if not (ASPECT_XZ in ("equal", "auto") or (isinstance(ASPECT_XZ, (int, float)) and ASPECT_XZ > 0.0)):
    parser.error('ASPECT_XZ must be "equal", "auto", or a positive number')

if YIDX is None and YOFF < 0:
    parser.error("--yoff must be >= 0")

for k, v in L_OVER.items():
    if v is not None and v <= 0.0:
        parser.error(f"{L_FLAG[k]} override must be positive")

if rank == 0:
    os.makedirs(OUTDIR, exist_ok=True)
comm.Barrier()

files = sorted(glob.glob(f"{BASE}/fields.*.bp"))
if rank == 0:
    print(f"\nFound {len(files)} files   (E_par mode = {EPAR_MODE})", flush=True)
    print(f"    Y slices   = {'indices ' + str(YIDX) if YIDX else f'Ny//2 +/- {YOFF} cells'}", flush=True)
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
    ascending, uniformly spaced coordinates; both are flagged if they do not hold. A --Lx/--Ly/--Lz
    override wins over the file, and is flagged if the two disagree by more than 1%.
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

def slab_xz(stream, name, j, Nz, Nx):
    """
    XZ plane at fixed y index j of a 3D array -> (Nz, Nx). Reads Nz*1*Nx elements, not Nz*Ny*Nx.
    Rows are already Z and columns X, so this goes straight into imshow with no transpose.
    """
    a = np.asarray(stream.read(name, start=[0, int(j), 0], count=[int(Nz), 1, int(Nx)]))
    return a.reshape(int(Nz), int(Nx))

def slice_fields(stream, j, Nz, Nx):
    """
    The four plotted quantities on one XZ slice, from the nine field components read on that
    slice only (3 E + 3 B + 3 J = 9 slabs, each Nz*Nx, never the full cube):

        J_z     = J3                                                  [n0 q0 c]
        E_par   = |(E.B)/|B||  (EPAR_MODE "abs", default; |B| guard)  [B0]
                   (E.B)/|B|   (EPAR_MODE "par",           |B| guard) [B0]
                    E.B        (EPAR_MODE "edotb", no guard needed)   [B0^2]
        E.J     = E1 J1 + E2 J2 + E3 J3                               [B0 n0 q0 c]
        B_x     = B1                                                  [B0]

    E.J is the TOTAL field-to-plasma power transfer: it contains the ideal (convective) part as
    well as the non-ideal one, so it lights up outflows as brightly as X-lines. Read it next to
    E_par, not instead of it.

    B_x is returned for the fourth panel: its zero contour locates the neutral surface within the
    slice, which is what tells you whether the two Y-cuts straddle the sheet.

    Returns (Jz, Epar, EJ, Bx, blanked_fraction). The fraction is reported so a slice that is
    mostly guard shows up in the log rather than silently blank.
    """
    E = [slab_xz(stream, n, j, Nz, Nx) for n in F_E]
    B = [slab_xz(stream, n, j, Nz, Nx) for n in F_B]
    J = [slab_xz(stream, n, j, Nz, Nx) for n in F_J]

    Jz    = J[2]
    Bx    = B[0]
    EdotJ = E[0] * J[0] + E[1] * J[1] + E[2] * J[2]
    EdotB = E[0] * B[0] + E[1] * B[1] + E[2] * B[2]

    if EPAR_MODE == "edotb":
        return Jz, EdotB, EdotJ, Bx, 0.0

    Bmag = np.sqrt(B[0] ** 2 + B[1] ** 2 + B[2] ** 2)
    med  = float(np.median(Bmag))
    #! fall back to the slice maximum if the median itself is ~0, so eps is never 0
    eps  = EPS_FRAC * (med if med > 0.0 else max(float(Bmag.max()), 1.0e-30))
    good = Bmag > eps
    Epar = np.full_like(EdotB, np.nan, dtype=float)
    np.divide(EdotB, Bmag, out=Epar, where=good)     #! leaves NaN where the guard fails
    if EPAR_MODE == "abs":
        #! the sign of E_par flips with the sign of B, i.e. with which side of the neutral surface
        #! the cell lies on, so the modulus is what carries the physics on a near-neutral cut
        Epar = np.abs(Epar)
    return Jz, Epar, EdotJ, Bx, float(1.0 - good.mean())

def sym_limits(data, fixed):
    """Symmetric limits about zero. NaN-safe, so the |B| guard does not poison the percentile."""
    if fixed is not None:
        v = abs(fixed)
    else:
        finite = np.abs(data[np.isfinite(data)])
        v = max(np.percentile(finite, PCT), 1.0e-30) if finite.size else 1.0
    return -v, v

def pos_limits(data, fixed):
    """[0, vmax] limits for a non-negative quantity (|E_par|). NaN-safe, same as sym_limits."""
    if fixed is not None:
        return 0.0, abs(fixed)
    finite = data[np.isfinite(data)]
    return 0.0, (max(np.percentile(finite, PCT), 1.0e-30) if finite.size else 1.0)

def draw(fig, ax, data, title, xlabel, ylabel, extent, cmap, vmin, vmax, aspect):
    #! a NUMBER is the box shape the panel should have, so the image fills the box ("auto") and
    #! set_box_aspect forces height/width; "equal" and "auto" are passed straight to imshow
    im = ax.imshow(data, origin="lower", aspect="auto" if _is_num(aspect) else aspect,
                   extent=extent, cmap=cmap, vmin=vmin, vmax=vmax)
    if _is_num(aspect):
        ax.set_box_aspect(float(aspect))
    ax.set_title(title, fontsize=14)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.tick_params(axis="both", labelsize=10)
    #! the colourbar is attached to the axes, never an inset: constrained_layout then reserves room for
    #! it AND for the next panel's tick labels, so nothing can overlap whatever the panel aspect is
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.ax.tick_params(labelsize=11)

def _is_num(a):
    return isinstance(a, (int, float)) and not isinstance(a, bool)

def row_height(aspect, data_ratio):
    """Height in inches of one row's plot box: data-true for "equal", PANEL for "auto", forced otherwise."""
    if aspect == "auto":
        return PANEL
    r = data_ratio if aspect == "equal" else float(aspect)
    return min(max(PANEL * r, 1.5), 2.0 * PANEL)     #! clamped so a very flat or tall box stays legible

def figure_size(rows, n_cols):
    """Figure size in inches. `rows` is one (aspect, Lvert/Lhoriz) pair per row."""
    return n_cols * (PANEL + PAD_W), sum(row_height(a, r) + PAD_H for a, r in rows)

LBL_X, LBL_Z = r"$x\ \omega_p/c$", r"$z\ \omega_p/c$"
EPAR_TITLE   = {"abs":   r"|E_\parallel|",
                "par":   r"E_\parallel",
                "edotb": r"\mathbf{E}\cdot\mathbf{B}"}[EPAR_MODE]
EPAR_CMAP    = CMAP_ABS if EPAR_MODE == "abs" else CMAP_SIGNED

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

        if z is None or Nz <= 1:
            #! an XZ plane needs x3; a 2D (x1, x2) run has none
            if rank == 0:
                print(f"ERROR: {fname} is 2D (no X3); an XZ plane does not exist in this run.", flush=True)
            break

        t_code = read_time(s)

        #! ---- the two Y slices: cell indices, clamped into the domain ----
        if YIDX is not None:
            j_top, j_bot = int(YIDX[0]), int(YIDX[1])
        else:
            j_top, j_bot = Ny // 2 + YOFF, Ny // 2 - YOFF
        for nm, j in (("top", j_top), ("bottom", j_bot)):
            if not (0 <= j < Ny):
                raise SystemExit(f"ERROR: {nm} slice index {j} is outside 0..{Ny - 1}")

        Jz_top, Ep_top, EJ_top, Bx_top, frac_top = slice_fields(s, j_top, Nz, Nx)
        Jz_bot, Ep_bot, EJ_bot, Bx_bot, frac_bot = slice_fields(s, j_bot, Nz, Nx)

    #! ---- per-axis spans; no assumption that Lx = Ly = Lz ----
    Lx, Ly, Lz = axis_span(x, "x1"), axis_span(y, "x2"), axis_span(z, "x3")
    ext = [0.0, Lx, 0.0, Lz]                     #! XZ plane: horizontal = X, vertical = Z

    #! physical y of each slice, for the panel titles (cell centres, pinned to the [0, Ly] axis)
    y_of = lambda j: (j + 0.5) * Ly / Ny

    if EPAR_MODE in ("abs", "par") and max(frac_top, frac_bot) > 0.02:
        warn_once("guard", f"|B| guard blanked up to {100 * max(frac_top, frac_bot):.1f}% of a slice; "
                           f"consider --epar edotb")

    #! ---- time in light-crossing times of the outflow direction ----
    time_label = (rf"$t\,c/L_x = {t_code / Lx:.2f}$" if np.isfinite(t_code) and Lx > 0
                  else f"step {step_idx:09d}")

    #! ---- 2 x 4: rows = the two Y slices, columns = J_z, E_par, E.J, B_x ----
    fig, axs = plt.subplots(2, 4, figsize=figure_size([(ASPECT_XZ, Lz / Lx)] * 2, 4),
                            constrained_layout=True)

    #! limits shared by BOTH rows, so the two slices are directly comparable.
    #! |E_par| is non-negative, so it gets [0, vmax] rather than a symmetric range.
    jz_lim = sym_limits(np.concatenate([Jz_top.ravel(), Jz_bot.ravel()]), JZ_VMAX)
    ej_lim = sym_limits(np.concatenate([EJ_top.ravel(), EJ_bot.ravel()]), EJ_VMAX)
    bx_lim = sym_limits(np.concatenate([Bx_top.ravel(), Bx_bot.ravel()]), BX_VMAX)
    ep_all = np.concatenate([Ep_top.ravel(), Ep_bot.ravel()])
    ep_lim = pos_limits(ep_all, EPAR_VMAX) if EPAR_MODE == "abs" else sym_limits(ep_all, EPAR_VMAX)

    rows = ((Jz_top, Ep_top, EJ_top, Bx_top, j_top), (Jz_bot, Ep_bot, EJ_bot, Bx_bot, j_bot))
    for row, (Jz_s, Ep_s, EJ_s, Bx_s, j) in enumerate(rows):
        ytag = rf"$y = {y_of(j):.2f}$  (cell {j})"
        draw(fig, axs[row, 0], Jz_s, rf"$J_z\ (X, Z)$,  {ytag}",
             LBL_X, LBL_Z, ext, CMAP_SIGNED, *jz_lim, ASPECT_XZ)
        draw(fig, axs[row, 1], Ep_s, rf"${EPAR_TITLE}\ (X, Z)$,  {ytag}",
             LBL_X, LBL_Z, ext, EPAR_CMAP, *ep_lim, ASPECT_XZ)
        draw(fig, axs[row, 2], EJ_s, rf"$\mathbf{{E}}\cdot\mathbf{{J}}\ (X, Z)$,  {ytag}",
             LBL_X, LBL_Z, ext, CMAP_SIGNED, *ej_lim, ASPECT_XZ)
        #! B_x last: its zero contour is where this plane crosses the neutral surface
        draw(fig, axs[row, 3], Bx_s, rf"$B_x\ (X, Z)$,  {ytag}",
             LBL_X, LBL_Z, ext, CMAP_SIGNED, *bx_lim, ASPECT_XZ)

    fig.suptitle(time_label, fontsize=18)
    outfile = f"{OUTDIR}/Jz_Epar_{step_idx:09d}.png"
    fig.savefig(outfile, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {outfile}", flush=True)

comm.Barrier()
if rank == 0:
    print("\nDone.", flush=True)