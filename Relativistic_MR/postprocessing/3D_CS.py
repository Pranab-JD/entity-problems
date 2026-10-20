"""
Created on Tue Oct 07 2026

@author: Pranab JD, Claude AI

3D rendering of the current sheet from a relativistic Harris-sheet Entity run: J_z (upper panel)
and rho (lower). Geometry (reversal across Y): outflow = X, inflow = Y, guide = Z.

Array axis order (Entity, 3D)
-----------------------------
    field arrays : (Nz, Ny, Nx)   [axis0 = x3, axis1 = x2, axis2 = x1]

Time is normalised by the light-crossing time of the outflow direction, t c / L_x.

Usage
-----
    sfolder="/scratch/project_465003132/RMR_pair_3D/sigma_10_Z100/"
    folder="/scratch/project_465003132/RMR_pair_3D/sigma_10_Z100/RMR/"
    fields="${folder}/fields"
    output="${folder}/plots"

    Lx=200.0; Ly=100.0; Lz=100.0

    #! pin the Y window so every frame spans the same Y and comes out the same size
    srun python3 -u ../postprocessing/Plot_CS_3D.py "$fields" "$output" \
        --Lx "$Lx" --Ly "$Ly" --Lz "$Lz" --ylim 35,65 --xz-aspect 5

    #! publication pass: sharper faces, higher dpi, a little more margin round the box
    srun python3 -u ../postprocessing/Plot_CS_3D.py "$fields" "$output" \
        --Lx "$Lx" --Ly "$Ly" --Lz "$Lz" --max-cells 720 --dpi 300 --pad-frac 0.12

User-modifiable settings
------------------------
Everything below is tunable. The CONSTANTS live in the USER SETTINGS block just under the imports
and are edited in the file; the FLAGS are command line and override nothing but themselves. Run
with -h for the one-line help on each flag.

  Flags -- input geometry
    --Lx --Ly --Lz           span overrides; None takes the span from the file
    --xz-aspect              X:Z ratio to DRAW at, or "true". Stretches a thin guide direction
    --y-aspect               drawn Y height as a fraction of L_x, or "true" for physical

  Flags -- sheet geometry measurement
    --wvar                   field the geometry is measured on: rho or jz
    --wfrac --efrac          peak fractions defining the thickness and the envelope
    --occ-min                occupied column fraction that marks the sheet as present at a Y
    --amp-min                how strong a column must be to contribute a thickness
    --smooth                 boxcar width in cells applied in Y and X before measuring
    --wref                   amplitude setting the thickness threshold: slab or column
    --prof-nz                number of Z slabs the geometry is measured on
    --diag                   also write the occupancy profile and thickness distribution

  Flags -- the drawn Y window
    --ylim                   FIX the window ("lo,hi" or "full"); overrides the three below
    --ywin                   extent (envelope + margin) or width (y_c +- wpad*thickness)
    --wpad                   margin beyond the envelope, in sheet thicknesses
    --mode                   full (both sides of the sheet) or centre (stop at the mid-plane)

  Flags -- rendering
    --slices                 see-through XZ planes through the window, or "auto"
    --slices-per-w           with auto, planes placed across one sheet thickness
    --slices-max             ceiling on the automatic count (cost is linear in it)
    --slice-cells            cell cap per side for those planes
    --alpha-max              opacity of the strongest cells
    --alpha-gamma            exponent on field strength that sets alpha
    --max-cells              cell cap per side on the walls
    --elev --azim            camera angles

  Flags -- output
    --dpi                    output resolution
    --panel-h                pin the panel height in inches
    --pad-frac               fractional margin added to every axis range
    --zoom                   box scale inside its axes; above 1.0 the corners clip
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")

from mpi4py import MPI
from adios2 import Stream

import argparse, os, glob
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.transforms import Bbox
from mpl_toolkits.mplot3d import Axes3D, proj3d    # noqa: F401  (Axes3D registers the projection)

#! ============================================================
#! USER SETTINGS
#! ============================================================
TIME_KEY = "Time"        #! adjust if Entity stores time under another key
F_JZ     = "fJ3"         #! out-of-plane current
F_N      = "fN"          #! number density

CMAP_JZ  = "seismic"
CMAP_RHO = "inferno"

JZ_VMAX  = None          #! symmetric about zero
RHO_VMIN = 0.0
RHO_VMAX = 8.0
PCT      = 99.0          #! percentile used wherever a limit above is None

#!   240 -> ~58k quads/face, a few seconds, a few hundred kB   (old default; visibly blocky)
#!   480 -> ~230k quads/face, ~10-30 s, a few MB               (default here)
#!   720 -> ~520k quads/face, ~1-2 min, ~10 MB                 (only if you really need it)
MAX_CELLS = 480

#! See-through stack. Drawing ONE opaque lid shows the sheet only where it cuts the box walls;
#! drawing SLICES parallel XZ planes whose alpha follows the field strength (see face_rgba) lets
#! the sheet be seen at every (X, Z). Cost is SLICES * SLICE_CELLS^2 quads, which is why the planes
#! get their own, coarser cell cap -- the stack carries the shape, the walls carry the detail.
#!   SLICES = 0 or 1 -> the old opaque single-lid rendering.
SLICES      = 9
SLICE_CELLS = 200
ALPHA_MAX   = 0.85       #! opacity of the strongest cells; < 1 so even those stay slightly see-through
ALPHA_GAMMA = 1.6        #! > 1 clears the faint upstream haze faster (alpha = strength**gamma)
BOX_EDGE_LW    = 0.6     #! wireframe round the volume in see-through mode; 0 switches it off
BOX_EDGE_COLOR = "0.55"

#! Panel geometry in inches. The two field panels are STACKED (J_z above, rho below) so each panel
#! gets the full figure width -- a 2:1 slab seen at azim 45 is a wide, shallow parallelogram and
#! wastes most of a square panel. Only the WIDTH is set here: the panel height follows the
#! projected shape of the box, so the figure never carries a band of white space above and below a
#! flat slab.
#!
#! TIGHTENED. The axis triad is gone (its orientation key is now the X / Z labels drawn on the
#! boxes themselves), so the strip it needed at the bottom is gone with it, and the remaining
#! bands have been cut to what the text in them actually occupies:
#!     TRIAD_H  1.30 -> BOTTOM_IN 0.10     (bare margin under the lower box)
#!     LABEL_IN 0.45 -> 0.28               (one line of 16 pt maths)
#!     TITLE_IN 0.60 -> 0.40               (one line of 20 pt maths)
#!     CB_IN    1.15 -> 0.92               (bar 0.26 + gap 0.16 + ~0.5 for the tick labels)
#!     MARGIN_IN 0.10 -> 0.05
#! That is ~1.9 inches of white space removed from a figure of about 9, at the same drawn box size.
#! If a colour bar's tick labels are ever clipped, CB_IN is the one to raise.
PANEL_W   = 11.0         #! drawing width of a panel
CB_IN     = 0.92         #! reserved strip on the right for the colour bar + its tick labels
LABEL_IN  = 0.28         #! band above each panel for its field label
TITLE_IN  = 0.40         #! band at the top for the time line
BOTTOM_IN = 0.10         #! bare margin below the lower panel (was the triad strip)
MARGIN_IN = 0.05         #! small left/right breathing space inside the panel width

#! Axis ticks along the two bottom-front edges of the box. The 3D axes themselves are switched off
#! (they would bring back panes and grids, and the panels overlap), so the ticks are drawn by hand
#! in data coordinates. TICK_N is the number of INTERVALS, so 5 gives 6 labels at
#! 0, Lx/5, 2Lx/5, 3Lx/5, 4Lx/5, Lx -- and likewise along Z.
TICK_N   = 5             #! intervals per axis -> TICK_N + 1 labelled ticks
TICK_FS  = 12            #! label font size
TICK_LEN = 0.030         #! mark length, as a fraction of the box extent it points along
TICK_PAD = 0.055         #! label offset from the edge, same units
TICK_COLOR = "0.25"

#! Axis NAMES, drawn beside the midpoint of the same two edges, further out than the tick numbers
#! so they do not collide with them. These replace the axis triad: each panel now says which way
#! X and Z run. AXIS_LABEL_PAD is in the same units as TICK_PAD (fraction of the extent the offset
#! points along) and must stay larger than it, or the name lands on top of a number. The padded
#! axis limits are widened to clear this offset (see `pad_h` below), so raising it does not push
#! the label off the panel.
AXIS_LABEL      = {"x": "X", "z": "Z"}   #! rename here, e.g. r"$x\,[c/\omega_p]$"
AXIS_LABEL_FS   = 16
AXIS_LABEL_PAD  = 0.125
AXIS_LABEL_COLOR = "0.10"

#! ZOOM scales the box inside its axes. Anything above ~1.0 pushes the far corners outside the
#! axes and CLIPS them -- that is what cut the left-top / right-bottom corners before. Margin is
#! now made with PAD_FRAC instead (see below), so the safe default is 1.0.
ZOOM = 1.0

#! Fractional margin added to every axis range. Because the SAME fraction is added to all three
#! axes, the ratio of the ranges -- and therefore the physical Lx : Lz : dY proportions set by
#! set_box_aspect -- is unchanged; the drawn box just sits inside its axes with room to spare, so
#! no corner can leave the panel at any --elev / --azim.
PAD_FRAC = 0.08

DPI = 200                #! raised from 120; with the wider panel this is ~2.7x the linear pixel
                         #! count of the old figure. Override with --dpi.

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

#! Optional span overrides. None -> take the span from the file's coordinate arrays.
parser.add_argument("--Lx", type=float, default=None)
parser.add_argument("--Ly", type=float, default=None)
parser.add_argument("--Lz", type=float, default=None)

#! --- sheet width ---
parser.add_argument("--wvar",    type=str,   default="rho", choices=["rho", "jz"],
                    help="field whose Y profile defines the sheet edges. Default: rho")
parser.add_argument("--wfrac",   type=float, default=0.5,
                    help="fraction of the peak above the baseline that marks the edge when the "
                         "THICKNESS is measured. 0.5 = FWHM. Default: 0.5")
parser.add_argument("--efrac",   type=float, default=0.25,
                    help="same, but for the ENVELOPE that sets the box. Lower than --wfrac so the "
                         "faint outer parts of a strongly kinked sheet stay inside the box. "
                         "Default: 0.25")
parser.add_argument("--occ-min",  type=float, default=0.01, dest="occ_min",
                    help="fraction of ALL columns that must be above threshold at a given Y for "
                         "the sheet to count as present there. This is what stops particle noise "
                         "from stretching the envelope to the domain edge; raise it if the box is "
                         "still too tall, lower it if a faint kink is being cut. Default: 0.01")
parser.add_argument("--amp-min",  type=float, default=0.25, dest="amp_min",
                    help="a column contributes a thickness only if its peak-above-baseline reaches "
                         "this fraction of the 90th-percentile amplitude in its slab. Rejects "
                         "columns with no real sheet, which otherwise report the width of a noise "
                         "bump. Default: 0.25")
parser.add_argument("--smooth",   type=int,   default=5,
                    help="boxcar width in CELLS applied in Y and X before the geometry is measured. "
                         "Without it the per-column peak is a particle-noise spike, which pushes "
                         "the half-max threshold up and shrinks every width. Keep it well below "
                         "the sheet thickness in cells. 1 disables. Default: 5")
parser.add_argument("--wref",     type=str,   default="slab", choices=["slab", "column"],
                    help="amplitude that sets the thickness threshold. 'slab': one reference for "
                         "the whole slab, so every column is measured at the same contour (steady "
                         "in time). 'column': each column's own peak, the textbook local FWHM, "
                         "noisier where plasmoids sit on the sheet. Default: slab")
parser.add_argument("--diag",    action="store_true",
                    help="also write CS_3D_diag_<step>.png: the occupancy profile and the "
                         "per-column thickness distribution the numbers were taken from")
parser.add_argument("--prof-nz", type=int,   default=16,
                    help="number of Z slabs the geometry is measured on. Raised from 5: a kink "
                         "that varies along Z needs several slabs before its envelope is captured. "
                         "Default: 16")
parser.add_argument("--ylim",    type=str,   default=None,
                    help='FIX the Y window instead of measuring it: "lo,hi" in code units, or '
                         '"full" for the whole domain. Every frame then spans the same Y, so a '
                         'sequence can be compared directly and all frames come out the same '
                         'size. Overrides --ywin, --wpad and --mode. A warning fires on any frame '
                         'whose sheet does not fit inside the fixed window.')
parser.add_argument("--ywin",    type=str,   default="extent", choices=["extent", "width"],
                    help="how the Y window is chosen. 'extent': the measured sheet envelope plus "
                         "a --wpad margin, so a large-amplitude kink stays fully inside. 'width': "
                         "the old y_c +- wpad*thickness rule. Default: extent")
parser.add_argument("--wpad",    type=float, default=1.5,
                    help="margin added beyond the envelope, in sheet THICKNESSES (or, with "
                         "--ywin width, the half-height of the box). Default: 1.5")

#! --- rendering ---
#! "full" (default) keeps the sheet mid-plane INSIDE the box: the Y window is symmetric about y_c.
#! "centre" is the old behaviour (box stops at y_c, mid-plane exposed on the top face).
parser.add_argument("--mode",     type=str,   default="full", choices=["full", "slab", "centre"],
                    help="Y window: 'full' (= 'slab') spans y_c +/- wpad*width so both sides of "
                         "the sheet are drawn; 'centre' stops the box at the mid-plane. "
                         "Default: full")
parser.add_argument("--elev",     type=float, default=30.0, help="view elevation [deg]. Default: 30")
parser.add_argument("--azim",     type=float, default=225.0, help="view azimuth in the XZ plane [deg]. Default: 225")
parser.add_argument("--y-aspect", type=str,   default="true", dest="y_aspect",
                    help='drawn height of the box as a fraction of L_x, or "true" for the '
                         'physical aspect ratio Lx:Lz:(y_top-y_bot). Default: true')
parser.add_argument("--xz-aspect", type=str,  default="true", dest="xz_aspect",
                    help='X:Z ratio the box is DRAWN at, or "true" for the physical Lx/Lz. Pass a '
                         'number to stretch the guide direction: a thin run (Lz = 10 against '
                         'Lx = 200, i.e. 20:1) leaves Z a sliver, and --xz-aspect 5 redraws it at '
                         '5:1 so structure along Z is visible. Z is then NOT to scale. '
                         'Default: true')

#! --- sharpness / cost ---
parser.add_argument("--max-cells", type=int, default=MAX_CELLS, dest="max_cells",
                    help=f"maximum drawn cells per face side (quad count goes like the square of "
                         f"this). Default: {MAX_CELLS}")
parser.add_argument("--dpi",       type=int, default=DPI,
                    help=f"output resolution. Default: {DPI}")
parser.add_argument("--zoom",      type=float, default=ZOOM,
                    help=f"scale of the box inside its axes. Above ~1.0 the far corners are "
                         f"CLIPPED; use --pad-frac for breathing room instead. "
                         f"Default: {ZOOM}")
parser.add_argument("--slices",      type=str,   default="auto",
                    help="number of see-through XZ planes stacked through the Y window, or 'auto' "
                         "to pick enough that the spacing resolves the sheet (see --slices-per-w). "
                         "Too few and a structure that varies in Y -- a flux rope lying along Z -- "
                         "is sampled as a few flat stripes instead of a tube. 0 or 1 gives the old "
                         "opaque lid. Default: auto")
parser.add_argument("--slices-per-w", type=float, default=4.0, dest="slices_per_w",
                    help="with --slices auto, how many planes to place across one sheet thickness. "
                         "Below about 3 the stack starts to alias. Default: 4")
parser.add_argument("--slices-max",  type=int,   default=48, dest="slices_max",
                    help="ceiling on the automatic count, because render time and file size scale "
                         "linearly with it. Default: 48")
parser.add_argument("--slice-cells", type=int,   default=SLICE_CELLS, dest="slice_cells",
                    help=f"cell cap per side for those planes (kept below --max-cells because the "
                         f"cost is multiplied by --slices). Default: {SLICE_CELLS}")
parser.add_argument("--alpha-max",   type=float, default=ALPHA_MAX, dest="alpha_max",
                    help=f"opacity of the strongest cells. Default: {ALPHA_MAX}")
parser.add_argument("--alpha-gamma", type=float, default=ALPHA_GAMMA, dest="alpha_gamma",
                    help=f"exponent on the field strength that sets alpha; raise it to clear the "
                         f"upstream haze, lower it to keep weak structure. Default: {ALPHA_GAMMA}")
parser.add_argument("--panel-h",   type=float, default=None, dest="panel_h",
                    help="drawing height of each panel in inches. Default: taken from the "
                         "projected shape of the box. PIN IT for a movie, so every frame comes "
                         "out the same pixel size.")
parser.add_argument("--pad-frac",  type=float, default=PAD_FRAC, dest="pad_frac",
                    help=f"fractional margin added to every axis range (applied equally, so the "
                         f"X:Y:Z proportions are unchanged). Raise it if a corner still touches "
                         f"the panel edge. Default: {PAD_FRAC}")
args = parser.parse_args()

#! CLI wins over the USER SETTINGS block above
MAX_CELLS, DPI, ZOOM, PAD_FRAC = args.max_cells, args.dpi, args.zoom, args.pad_frac
#! "auto" -> decided per frame, once the Y window and the thickness are known
SLICES = "auto" if str(args.slices).strip().lower() == "auto" else max(int(args.slices), 0)
SLICE_CELLS = args.slice_cells
if args.slices_per_w <= 0.0:
    parser.error("--slices-per-w must be positive")
if args.slices_max < 2:
    parser.error("--slices-max must be >= 2")
ALPHA_MAX, ALPHA_GAMMA = args.alpha_max, args.alpha_gamma
if SLICE_CELLS < 16:
    parser.error("--slice-cells must be >= 16")
if not (0.0 < ALPHA_MAX <= 1.0):
    parser.error("--alpha-max must lie in (0, 1]")
if ALPHA_GAMMA <= 0.0:
    parser.error("--alpha-gamma must be positive")
if MAX_CELLS < 16:
    parser.error("--max-cells must be >= 16")
if DPI < 50:
    parser.error("--dpi must be >= 50")
if ZOOM <= 0.0:
    parser.error("--zoom must be positive")
if PAD_FRAC < 0.0:
    parser.error("--pad-frac must be >= 0")
if args.panel_h is not None and args.panel_h <= 0.0:
    parser.error("--panel-h must be positive")
if ZOOM > 1.0:
    warn_only_zoom = True            #! warned once below, after warn_once() is defined
else:
    warn_only_zoom = False

#! backwards compatibility: "slab" and "full" are the same symmetric window
MODE = "full" if args.mode in ("full", "slab") else "centre"

BASE, OUTDIR = args.base, args.outdir
L_OVER = {"x1": args.Lx, "x2": args.Ly, "x3": args.Lz}      #! None = use the file
L_FLAG = {"x1": "--Lx", "x2": "--Ly", "x3": "--Lz"}         #! for the mismatch message

if not (0.0 < args.wfrac < 1.0):
    parser.error("--wfrac must lie strictly between 0 and 1")
if not (0.0 < args.efrac < 1.0):
    parser.error("--efrac must lie strictly between 0 and 1")
if args.prof_nz < 1:
    parser.error("--prof-nz must be >= 1")
if args.wpad <= 0.0:
    parser.error("--wpad must be positive")
if not (0.0 < args.occ_min <= 1.0):
    parser.error("--occ-min must lie in (0, 1]")
if not (0.0 <= args.amp_min < 1.0):
    parser.error("--amp-min must lie in [0, 1)")
if args.smooth < 1:
    parser.error("--smooth must be >= 1")
YWIN, OCC_MIN = args.ywin, args.occ_min

#! --ylim: None (measure per frame), "full" (whole domain), or an explicit (lo, hi)
YLIM = None
if args.ylim is not None:
    if args.ylim.strip().lower() == "full":
        YLIM = "full"
    else:
        try:
            lo_s, hi_s = args.ylim.replace(":", ",").split(",")
            YLIM = (float(lo_s), float(hi_s))
        except ValueError:
            parser.error('--ylim must be "lo,hi" or "full"')
        if YLIM[1] <= YLIM[0]:
            parser.error("--ylim needs hi > lo")
if args.efrac > args.wfrac:
    warn_only_efrac = True            #! warned once below, after warn_once() is defined
else:
    warn_only_efrac = False

Y_ASPECT = None if args.y_aspect.lower() == "true" else float(args.y_aspect)
if Y_ASPECT is not None and Y_ASPECT <= 0.0:
    parser.error('--y-aspect must be positive or "true"')

XZ_ASPECT = None if args.xz_aspect.lower() == "true" else float(args.xz_aspect)
if XZ_ASPECT is not None and XZ_ASPECT <= 0.0:
    parser.error('--xz-aspect must be positive or "true"')

for k, v in L_OVER.items():
    if v is not None and v <= 0.0:
        parser.error(f"{L_FLAG[k]} override must be positive")

if rank == 0:
    os.makedirs(OUTDIR, exist_ok=True)
comm.Barrier()

files = sorted(glob.glob(f"{BASE}/fields.*.bp"))
if rank == 0:
    print(f"\nFound {len(files)} files", flush=True)
    print(f"    L override = {{x1: {args.Lx}, x2: {args.Ly}, x3: {args.Lz}}}   (None -> read from the file)", flush=True)
    print(f"    geometry from {args.wvar} on {args.prof_nz} Z slabs, column by column: "
          f"thickness at {args.wfrac:g} of peak (amp-min {args.amp_min:g}), "
          f"envelope at {args.efrac:g} with occupancy >= {OCC_MIN:g}, "
          f"y-window = {'--ylim ' + str(args.ylim) if YLIM is not None else YWIN}", flush=True)
    print(f"    mode = {MODE}, view = (elev {args.elev:g}, azim {args.azim:g}), "
          f"y-aspect = {args.y_aspect}, xz-aspect = {args.xz_aspect}", flush=True)
    print(f"    max cells/side = {MAX_CELLS}, dpi = {DPI}, zoom = {ZOOM:g}, "
          f"pad-frac = {PAD_FRAC:g}", flush=True)
    print(f"    slices = {SLICES}"
          f"{f' (<= {args.slices_max}, {args.slices_per_w:g}/thickness)' if SLICES == 'auto' else ''}"
          f" at {SLICE_CELLS} cells/side, "
          f"alpha max = {ALPHA_MAX:g}, gamma = {ALPHA_GAMMA:g}\n", flush=True)
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

#! a zoom above 1 is exactly what clipped the corners before -- say so rather than silently obeying
if warn_only_zoom:
    warn_once("zoom_gt_1", f"--zoom {ZOOM:g} > 1: the box is drawn larger than its axes and the "
                           f"far corners will be cut off; raise --pad-frac instead")
#! the envelope should be measured at a LOWER threshold than the thickness, or the box can end up
#! narrower than the structure it is meant to contain
if warn_only_efrac:
    warn_once("efrac_gt_wfrac",
              f"--efrac {args.efrac:g} > --wfrac {args.wfrac:g}: the envelope is being measured at "
              f"a higher threshold than the thickness, so the box may clip the sheet")
#! the axis NAME has to sit outside the tick numbers, or the two overprint each other
if AXIS_LABEL_PAD <= TICK_PAD:
    warn_once("axlbl_pad",
              f"AXIS_LABEL_PAD ({AXIS_LABEL_PAD:g}) is not larger than TICK_PAD ({TICK_PAD:g}): "
              f"the X / Z names will overlap the tick numbers")

def axis_span(coord, name):
    """
    Span of one axis from its cell CENTRES: X1/X2/X3 have the same length as the field axis, so the
    domain runs from c[0] - dx/2 to c[-1] + dx/2. Entity writes ascending, uniformly spaced
    coordinates; both are flagged if they do not hold. A --Lx/--Ly/--Lz override wins over the
    file, and is flagged if the two disagree by more than 1%.
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

def domain_edges(c):
    """
    Outer CELL EDGES of an axis given its cell centres: (c[0] - dc/2, c[-1] + dc/2).
    Used so the drawn faces cover the WHOLE domain and not just centre-to-centre.
    """
    c  = np.asarray(c, dtype=float).ravel()
    dc = float(np.median(np.diff(c))) if c.size > 1 else 1.0
    return float(c[0] - 0.5 * dc), float(c[-1] + 0.5 * dc)

#! ---- hyperslab reads: every face is 2D, so no full (Nz, Ny, Nx) cube is ever in memory ----
def face_xz(stream, name, j, Nz, Nx):
    """Horizontal plane at one Y index -> (Nz, Nx): rows = Z, columns = X."""
    a = np.asarray(stream.read(name, start=[0, int(j), 0], count=[int(Nz), 1, int(Nx)]))
    return a.reshape(int(Nz), int(Nx))

def face_zy(stream, name, i, Nz, j0, nj):
    """Vertical plane at one X index, Y window [j0, j0+nj) -> (nj, Nz): rows = Y, columns = Z."""
    a = np.asarray(stream.read(name, start=[0, int(j0), int(i)], count=[int(Nz), int(nj), 1]))
    return a.reshape(int(Nz), int(nj)).T

def face_xy(stream, name, k, Nx, j0, nj):
    """Vertical plane at one Z index, Y window [j0, j0+nj) -> (nj, Nx): rows = Y, columns = X."""
    a = np.asarray(stream.read(name, start=[int(k), int(j0), 0], count=[1, int(nj), int(Nx)]))
    return a.reshape(int(nj), int(Nx))

def slab_xy_full(stream, name, k, Ny, Nx):
    """Full-Y plane at one Z index -> (Ny, Nx). Used only for the width profiles."""
    a = np.asarray(stream.read(name, start=[int(k), 0, 0], count=[1, int(Ny), int(Nx)]))
    return a.reshape(int(Ny), int(Nx))

#! ---- sheet geometry: EXTENT and THICKNESS are two different things ----
#!
#! The old code measured one X-AVERAGED profile per Z slab and used its half-maximum width for
#! BOTH the box height and the quoted Delta_CS. Both uses break once the sheet kinks by more than
#! its own thickness:
#!
#!   * the box.   Averaging over X first collapses the kink: a sheet that runs from y = 40 at one
#!                X to y = 60 at another averages into a broad, low bump whose half-max crossings
#!                sit nowhere near +-wpad of the true excursion. The window then lands off-centre
#!                and clips one side of the sheet -- the top half missing in one snapshot, the
#!                bottom half in the next, which is exactly the failure in the first two figures.
#!   * the label. That same smearing INFLATES the quoted width: the 19.88 c/omega_p in the third
#!                figure is mostly kink amplitude, not sheet thickness.
#!
#! So they are now measured separately, COLUMN BY COLUMN (one Y profile per (x, z) pair), which is
#! immune to the kink because each column sees the sheet at its own local Y position:
#!
#!   extent    where the sheet is PRESENT, judged from the fraction of columns above threshold at
#!             each Y (the occupancy profile). This sets the box.
#!   thickness the MEDIAN over columns of the per-column `wfrac` width. A local measurement, so it
#!             stays a thickness however violently the sheet meanders. This sets the label.
#!
#! Both are read off the same slabs, so the extra cost is arithmetic, not I/O.
#!
#! SECOND REVISION. The first attempt took each column's OUTERMOST crossings -- first and last cell
#! above threshold -- for both quantities. On clean synthetic data that works; on a real PIC run it
#! collapses, because a single stray cell anywhere in a column moves the edge all the way out:
#!   * the envelope saturated to the whole domain on every snapshot ([0.6, 124.4] in a 125-deep
#!     box), since over ~10^5 columns some cell is always above threshold near the boundary;
#!   * the thickness swung between 4 and 15 with a 16-84% spread of 0.9-14, i.e. the per-column
#!     numbers were noise, not a measurement -- a column whose outermost crossings straddle a
#!     plasmoid, the far side of a fold, or one noisy cell reports a "width" of tens of d_e.
#! Both are now robust statistics instead of extreme-value ones:
#!   * thickness uses the CONTIGUOUS run above threshold THROUGH THE COLUMN PEAK, so a distant
#!     stray cell cannot contribute, and only columns carrying a real sheet are counted;
#!   * the envelope uses the occupancy profile with a minimum occupied FRACTION, so moving it
#!     requires a sizeable population of columns, not one outlier.
def boxcar(a, n, axis):
    """
    Running mean of width n along `axis`, with shrinking windows at the ends so the array keeps its
    length and no edge is biased towards zero. Cumulative-sum implementation: O(N), no scipy.
    """
    n = int(n)
    if n <= 1:
        return a
    a   = np.asarray(a, dtype=float)
    N   = a.shape[axis]
    h   = n // 2
    c   = np.cumsum(np.moveaxis(a, axis, 0), axis=0)
    c   = np.concatenate([np.zeros((1,) + c.shape[1:]), c], axis=0)
    lo  = np.clip(np.arange(N) - h,         0, N)
    hi  = np.clip(np.arange(N) - h + n,     0, N)
    cnt = (hi - lo).astype(float).reshape((-1,) + (1,) * (c.ndim - 1))
    return np.moveaxis((c[hi] - c[lo]) / cnt, 0, axis)

def column_stats(P, y, efrac, wfrac, amp_min, smooth, wref):
    """
    Per-column sheet geometry for one XY slab P (Ny, Nx), columns along X.

    Returns (width, occ, ncol, nvalid):
      width   (Nx,)  contiguous `wfrac` width about each column's peak; NaN where the column does
                     not carry a usable sheet
      occ     (Ny,)  number of columns above their own `efrac` threshold at each Y -- accumulate
                     this over slabs and divide by `ncol` to get the occupied fraction
      ncol    int    columns contributing to `occ`
      nvalid  int    columns that passed the amplitude test and so contributed a width

    A column is usable when its own peak-above-baseline rises to at least `amp_min` of the 90th
    percentile of that amplitude over the slab. Without this test, columns where the sheet is
    locally absent or depleted (an X-point, an outflow gap) report the width of whatever noise
    bump happened to be tallest -- which is where the spuriously SMALL widths came from.

    Crossings are interpolated linearly between the bracketing cells; taking the outermost cell
    CENTRE instead biases every width low by up to one cell per side.

    `smooth` is a boxcar width in CELLS applied in both Y and X first, and it is not cosmetic: on
    raw PIC data the per-column maximum is a NOISE spike, not the sheet peak, so the half-maximum
    threshold sits far too high, and the contiguous run above it then breaks apart on cell-scale
    fluctuations. Both effects shrink the measured width, and both get worse as the run gets
    noisier -- which is why the thickness appeared to drift with time rather than hold steady.
    Keep `smooth` well below the sheet thickness in cells and it costs nothing in resolution.
    """
    P  = np.asarray(P, dtype=float)
    if smooth > 1:
        P = boxcar(boxcar(P, smooth, axis=0), smooth, axis=1)
    Ny, Ncol = P.shape
    m  = max(int(0.1 * Ny), 1)
    #! baseline per column, from the outer 10% at each end: a sheet on a finite background is
    #! measured against that background, not against zero
    base = 0.5 * (np.median(P[:m], axis=0) + np.median(P[-m:], axis=0))
    peak = np.nanmax(P, axis=0)
    amp  = peak - base
    good = np.isfinite(amp) & (amp > 0.0)
    #! reference amplitude: 90th percentile, not the median, so a slab where many columns are
    #! sheet-free still compares against the real sheet rather than against the noise floor
    ref   = np.nanpercentile(amp[good], 90.0) if good.any() else 0.0
    valid = good & (amp >= amp_min * ref) & (ref > 0.0)

    #! ---- occupancy: how many columns are above their own efrac threshold at each Y ----
    occ = (P > (base + efrac * np.where(good, amp, np.inf))[None, :]).sum(axis=1)

    #! ---- thickness: contiguous run above the wfrac threshold THROUGH the column peak ----
    #! Which amplitude sets that threshold matters more than it looks:
    #!   "column"  each column's own peak -> the textbook local FWHM, but a plasmoid sitting on the
    #!             sheet raises that column's peak and so its threshold, and the width measured at
    #!             the higher contour comes out narrower. Across a plasmoid-rich sheet this adds
    #!             scatter that has nothing to do with the sheet thickening.
    #!   "slab"    one reference amplitude for the whole slab (the median over usable columns), so
    #!             every column is measured at the SAME density contour. Steadier in time, which is
    #!             what a thickness-vs-time curve needs; the default.
    #! Neither is wrong -- they answer slightly different questions. --diag shows the distribution.
    amp_w = amp if wref == "column" else np.full_like(amp, float(np.median(amp[valid]))
                                                      if valid.any() else 1.0)
    thr  = base + wfrac * np.where(good, amp_w, np.inf)
    mask = P > thr[None, :]
    pk   = np.argmax(np.where(np.isfinite(P), P, -np.inf), axis=0)   #! peak row of each column
    cols = np.arange(Ncol)
    rows = np.arange(Ny)[:, None]

    #! last row BELOW threshold at or under r  ->  the contiguous run starts just after it
    idx_lo = np.where(mask, -1, rows)
    c_lo   = np.maximum.accumulate(idx_lo, axis=0)
    c_lo   = np.vstack([np.full((1, Ncol), -1), c_lo])              #! shift: strictly below r
    lo_i   = c_lo[pk, cols] + 1

    #! first row BELOW threshold at or above r  ->  the run ends just before it
    idx_hi = np.where(mask, Ny, rows)
    c_hi   = np.minimum.accumulate(idx_hi[::-1], axis=0)[::-1]
    c_hi   = np.vstack([c_hi, np.full((1, Ncol), Ny)])              #! shift: strictly above r
    hi_i   = c_hi[pk + 1, cols] - 1

    #! sub-cell crossings on the two bracketing pairs
    jlo  = np.maximum(lo_i - 1, 0)
    f    = P[lo_i, cols] - P[jlo, cols]
    f    = np.where(np.abs(f) < 1.0e-30, 1.0, f)
    wlo  = np.where(lo_i > 0, (thr - P[jlo, cols]) / f, 0.0)
    y_lo = y[jlo] + np.clip(wlo, 0.0, 1.0) * (y[lo_i] - y[jlo])

    jhi  = np.minimum(hi_i + 1, Ny - 1)
    g    = P[hi_i, cols] - P[jhi, cols]
    g    = np.where(np.abs(g) < 1.0e-30, 1.0, g)
    whi  = np.where(hi_i < Ny - 1, (P[hi_i, cols] - thr) / g, 0.0)
    y_hi = y[hi_i] + np.clip(whi, 0.0, 1.0) * (y[jhi] - y[hi_i])

    width = np.where(valid, y_hi - y_lo, np.nan)
    return width, occ, Ncol, int(valid.sum())

#! NOTE: the old gradient-based cross-check (steepest rise to steepest fall of the X-AVERAGED
#! profile) has been dropped. On a kinked sheet that profile is the kink envelope, not the sheet,
#! so the number it returned was a kink amplitude wearing a thickness label. The spread of the
#! per-column widths is reported instead -- it says how uniform the thickness really is.

#! ---- downsampling ----
#! CHANGED: the block means now KEEP the partial trailing block instead of discarding it, so the
#! last (up to stride-1) cells in X, Y and Z still appear on the drawn faces. Previously those
#! cells were dropped while the coordinate axis was stretched to the full span, which both lost
#! data at the far X/Z edges and shifted everything by up to half a block.
def stride_for(n, cap=None):
    return max(1, int(np.ceil(n / float(MAX_CELLS if cap is None else cap))))

def _reduce_mean(a, s, axis):
    """Mean over blocks of s along `axis`, with a short final block if s does not divide the length."""
    if s == 1:
        return a
    n   = a.shape[axis]
    idx = np.arange(0, n, s)                                  #! block start indices
    cnt = np.diff(np.append(idx, n)).astype(float)            #! true length of each block (last may be short)
    out = np.add.reduceat(a, idx, axis=axis)
    shp = [1] * out.ndim
    shp[axis] = cnt.size
    return out / cnt.reshape(shp)

def block_mean(a, sr, sc):
    """Mean over sr x sc blocks; no cell is dropped (see _reduce_mean)."""
    return _reduce_mean(_reduce_mean(np.asarray(a, dtype=float), sr, 0), sc, 1)

def block_mean_1d(c, s):
    return _reduce_mean(np.asarray(c, dtype=float)[:, None], s, 0)[:, 0]

def n_blocks(n, s):
    """Number of blocks (including the short trailing one) that block_mean will produce."""
    return int(np.ceil(n / float(s)))

#! ---- rendering ----
def all_faces(F):
    """Every 2D array a panel will draw, as a flat list (planes + walls + optional bottom)."""
    out = list(F["planes"]) + [F["xw"], F["zw"]]
    if F["bot"] is not None:
        out.append(F["bot"])
    return out

def limits(kind, faces):
    """Colour limits from the drawn faces, honouring any fixed value in USER SETTINGS."""
    allv = np.concatenate([f.ravel() for f in faces if f is not None])
    allv = allv[np.isfinite(allv)]
    if kind == "jz":
        v = abs(JZ_VMAX) if JZ_VMAX is not None else max(float(np.percentile(np.abs(allv), PCT)), 1.0e-30)
        return -v, v
    vmin = RHO_VMIN if RHO_VMIN is not None else float(np.percentile(allv, 100.0 - PCT))
    vmax = RHO_VMAX if RHO_VMAX is not None else float(np.percentile(allv, PCT))
    return vmin, max(vmax, vmin + 1.0e-30)

def set_box(ax, box, zoom):
    """set_box_aspect with the zoom argument where it exists (matplotlib >= 3.6), without it otherwise."""
    try:
        ax.set_box_aspect(box, zoom=zoom)
    except TypeError:
        ax.set_box_aspect(box)
        warn_once("nozoom", "this matplotlib has no zoom argument for set_box_aspect; "
                            "the panels will carry extra white space")

def face_rgba(cmap, norm, data, kind, see_through):
    """
    Per-cell RGBA for one face.

    With see_through on, the ALPHA of each cell is set from how far its value sits from the
    "nothing here" end of the colour scale: the centre for the diverging J_z map, the bottom for
    the sequential rho map. Upstream plasma therefore becomes nearly invisible and the sheet keeps
    its opacity, so stacked planes let the current sheet be seen at EVERY (X, Z), not only where it
    cuts the box walls. ALPHA_GAMMA > 1 clears the faint haze faster; ALPHA_MAX caps how solid the
    strongest cells get, so even those do not completely hide what is behind them.
    """
    t    = np.asarray(norm(data), dtype=float)        #! 0..1 within the colour limits
    rgba = cmap(t)
    if not see_through:
        return rgba
    m = np.abs(2.0 * t - 1.0) if kind == "jz" else t  #! diverging: distance from the centre
    rgba[..., 3] = np.clip(np.nan_to_num(m), 0.0, 1.0) ** ALPHA_GAMMA * ALPHA_MAX
    return rgba

def surf(ax, MX, MY, MZ, rgba):
    """
    One coloured face. The grids MX/MY/MZ are CELL EDGES with shape (n+1, m+1) and `rgba` is the
    (n, m, 4) cell colour array, so plot_surface paints exactly n*m quads and nothing is clipped at
    the far edge -- passing centre grids of the same shape as the data silently drops the last
    row/column. shade=False keeps the colours quantitative; rstride/cstride = 1 because the arrays
    are already block-averaged. antialiased=True smooths the quad outlines (the old
    antialiased=False left a hard stair-step on every edge, which is most of what read as
    "pixelated"); edgecolor "none" plus linewidth 0 keeps the quads from being outlined in their
    own colour. No `alpha=` is passed, so the alpha channel inside `rgba` is what counts.
    """
    ax.plot_surface(MX, MY, MZ, facecolors=rgba, rstride=1, cstride=1,
                    shade=False, linewidth=0.0, edgecolor="none", antialiased=True)

def setup_axes3d(ax, geom):
    """Limits, physical box aspect, view and a bare axes -- the same for every panel and probe."""
    ax.set_xlim(geom["xlim"])        #! mpl-x = physical X
    ax.set_ylim(geom["zlim"])        #! mpl-y = physical Z
    ax.set_zlim(geom["ylim"])        #! mpl-z = physical Y
    set_box(ax, geom["box"], ZOOM)
    ax.view_init(elev=args.elev, azim=args.azim)
    ax.set_axis_off()                #! requirement: no grid, no panes, no ticks
    #! The fitted rectangle is larger than its slot (that is how the box ends up filling the slot),
    #! so the panels OVERLAP. An opaque axes background would then let the lower panel paint over
    #! the bottom of the upper one -- which is what cut the upper box off. Make it transparent.
    ax.patch.set_visible(False)

#! ---- fitting the box to its panel ----
#! The old figure guessed a --zoom to fill the panel, which either wasted half the panel or pushed
#! the far corners outside it (the clipped left-top / right-bottom corners). Instead the eight
#! corners of the limits box are PROJECTED for the current view, and the panel is sized and placed
#! so that projection fits exactly. The projection is measured on an EMPTY probe figure, so the
#! quads are rendered only once; the shape of the projected box does not depend on the axes
#! rectangle (set_box_aspect preserves it), only its scale does, which is what the fit solves for.
def _ink_bbox(fig, ax, geom):
    """Figure-coordinate bbox of the projected corners of the limits box."""
    M   = ax.get_proj()
    pts = np.array([proj3d.proj_transform(a, b, c, M)[:2]
                    for a in geom["xlim"] for b in geom["zlim"] for c in geom["ylim"]])
    f = fig.transFigure.inverted().transform(ax.transData.transform(pts))
    return Bbox.from_extents(f[:, 0].min(), f[:, 1].min(), f[:, 0].max(), f[:, 1].max())

def probe_ink(figsize, rect, geom):
    """Ink bbox an empty 3D axes of this figsize/rect would produce. Cheap: nothing is drawn."""
    f = plt.figure(figsize=figsize)
    a = f.add_axes(rect, projection="3d")
    setup_axes3d(a, geom)
    f.canvas.draw()                  #! Axes3D fixes its 2D view limits at draw time, so this is needed
    bb = _ink_bbox(f, a, geom)
    plt.close(f)
    return bb

#! ---- cache the projection: geom only changes when the Y window does ----
_ink_cache = {}

def ink_aspect(geom):
    """Height/width of the projected box, in inches. Independent of the axes rectangle."""
    #! key on the three limit tuples -- the only part of geom the projection depends on
    key = (geom["xlim"], geom["zlim"], geom["ylim"], args.elev, args.azim)
    if key in _ink_cache:
        return _ink_cache[key]
    probe = probe_ink((10.0, 10.0), [0.05, 0.05, 0.90, 0.90], geom)
    w = max(probe.width, 1.0e-12)
    _ink_cache[key] = float(probe.height / w)
    return _ink_cache[key]

def fitted_rect(figsize, slot, geom, bind="width"):
    """
    Axes rectangle whose projected box fills `slot` (a Bbox in figure coordinates). The rectangle
    keeps the slot's shape and is only scaled and shifted, so the projected box scales uniformly
    and cannot be distorted. Margin comes from PAD_FRAC on the limits, not here.

    bind="width" (default) scales so the projected box fills the slot WIDTH, whatever its height.
    This is what keeps the X extent of the drawing CONSTANT from frame to frame: the projected
    width depends on Lx and Lz and the view angles, none of which change, while the height follows
    the Y window, which does. Binding on min(width, height) instead -- the previous behaviour --
    let a tall Y window steal scale from X, so a snapshot with a thick sheet came out visibly
    narrower than one with a thin sheet. The slot height is derived from the same projection (see
    ink_aspect below), so nothing overflows.

    bind="fit" restores the min() rule, needed when --panel-h pins a height the projection did not
    choose and the drawing would otherwise run past its slot.
    """
    ink = probe_ink(figsize, [slot.x0, slot.y0, slot.width, slot.height], geom)
    sw  = slot.width / max(ink.width, 1.0e-12)
    s   = sw if bind == "width" else min(sw, slot.height / max(ink.height, 1.0e-12))
    nw, nh = slot.width * s, slot.height * s
    rx = (ink.x0 + 0.5 * ink.width  - slot.x0) / slot.width      #! ink centre, as a fraction of the rect
    ry = (ink.y0 + 0.5 * ink.height - slot.y0) / slot.height
    return [slot.x0 + 0.5 * slot.width  - rx * nw,
            slot.y0 + 0.5 * slot.height - ry * nh, nw, nh]

def draw_ticks(ax, geom):
    """
    Ticks, value labels AND the axis names along the two bottom edges of the box that face the
    camera: one running in X, one in Z. The 3D axes are off, so these are ordinary 3D line and text
    artists placed in data coordinates; they sit just outside the box, which is why PAD_FRAC has to
    leave room for them (see `padded` below). Which edges are used follows the same azimuth test as
    the walls, so the ticks stay in front whatever --azim is.

    The axis NAMES (CHANGED: these are new, and replace the axis triad that used to sit under the
    figure) go at the midpoint of the same edge, offset by AXIS_LABEL_PAD -- further out than
    TICK_PAD so they clear the numbers. They are placed in DATA coordinates like everything else
    here, so they follow the box under any --elev / --azim and shrink and grow with it; the cost is
    that they are not rotated into the plane of the edge, which 3D text in matplotlib does not do
    without a manual projection.
    """
    (x0, x1), (z0, z1), (y0, _) = geom["box_x"], geom["box_z"], geom["box_y"]
    dx, dz = x1 - x0, z1 - z0
    #! point the marks away from the box, on the side the camera is on
    outz = 1.0 if geom["z_wall"] >= 0.5 * (z0 + z1) else -1.0
    outx = 1.0 if geom["x_wall"] >= 0.5 * (x0 + x1) else -1.0
    fx = "%.0f" if dx >= 10.0 else "%.2g"
    fz = "%.0f" if dz >= 10.0 else "%.2g"

    #! --- ticks along X, drawn on the near-Z edge ---
    for f in np.linspace(0.0, 1.0, TICK_N + 1):
        xv = x0 + f * dx
        ax.plot([xv, xv], [geom["z_wall"], geom["z_wall"] + outz * TICK_LEN * dz], [y0, y0],
                color=TICK_COLOR, lw=0.9, zorder=5)
        ax.text(xv, geom["z_wall"] + outz * TICK_PAD * dz, y0, fx % xv,
                color=TICK_COLOR, fontsize=TICK_FS, ha="center", va="center", zorder=5)

    #! --- ticks along Z, drawn on the near-X edge ---
    for f in np.linspace(0.0, 1.0, TICK_N + 1):
        zv = z0 + f * dz
        ax.plot([geom["x_wall"], geom["x_wall"] + outx * TICK_LEN * dx], [zv, zv], [y0, y0],
                color=TICK_COLOR, lw=0.9, zorder=5)
        ax.text(geom["x_wall"] + outx * TICK_PAD * dx, zv, y0, fz % zv,
                color=TICK_COLOR, fontsize=TICK_FS, ha="center", va="center", zorder=5)

    #! --- axis names, at the midpoint of each of those two edges ---
    ax.text(0.5 * (x0 + x1), geom["z_wall"] + outz * AXIS_LABEL_PAD * dz, y0, AXIS_LABEL["x"],
            color=AXIS_LABEL_COLOR, fontsize=AXIS_LABEL_FS, fontweight="bold",
            ha="center", va="center", zorder=6)
    ax.text(geom["x_wall"] + outx * AXIS_LABEL_PAD * dx, 0.5 * (z0 + z1), y0, AXIS_LABEL["z"],
            color=AXIS_LABEL_COLOR, fontsize=AXIS_LABEL_FS, fontweight="bold",
            ha="center", va="center", zorder=6)

def render(fig, ax, kind, F, geom, cax):
    """
    F is the face dict for one field: "planes" (XZ planes, bottom to top, with their Y positions in
    geom["plane_y"]), "xw" / "zw" (the two camera-facing walls) and "bot" (optional bottom face).
    geom carries the EDGE grids and the box extents. `cax` is the axes the colour bar is drawn in
    (placed by the caller, not stolen from `ax`, so the panel keeps the full slot it was fitted to).

    With SLICES > 1 the panel is a stack of see-through XZ planes rather than one opaque lid: each
    plane hides almost nothing where the plasma is upstream, so the sheet shows through from any
    (X, Z). Planes are drawn bottom-first; matplotlib then depth-sorts the collections, which is
    reliable here because parallel planes never interpenetrate.
    """
    cmap = plt.get_cmap(CMAP_JZ if kind == "jz" else CMAP_RHO)
    norm = Normalize(*limits(kind, all_faces(F)))
    see  = len(F["planes"]) > 1             #! a single plane stays opaque -- nothing to see through

    for y_val, dat in zip(geom["plane_y"], F["planes"]):
        surf(ax, geom["PX"], geom["PZ"], np.full_like(geom["PX"], y_val),
             face_rgba(cmap, norm, dat, kind, see))
    #! near X wall (ZY) and near Z wall (XY), spanning the whole Y window. They are see-through on
    #! the same rule, otherwise the near walls would be an opaque screen in front of the stack.
    surf(ax, np.full_like(geom["WZ"], geom["x_wall"]), geom["WZ"], geom["WY_zy"],
         face_rgba(cmap, norm, F["xw"], kind, see))
    surf(ax, geom["WX"], np.full_like(geom["WX"], geom["z_wall"]), geom["WY_xy"],
         face_rgba(cmap, norm, F["zw"], kind, see))
    #! bottom face only when the camera is below the slab (otherwise hidden, and risks z-order mess)
    if F["bot"] is not None:
        surf(ax, geom["PX"], geom["PZ"], np.full_like(geom["PX"], geom["y_bot"]),
             face_rgba(cmap, norm, F["bot"], kind, see))

    #! With see-through faces the slab has no silhouette left, so outline the volume: a thin
    #! wireframe of the 12 box edges keeps the rendering readable as a 3D box rather than a sheet
    #! floating in space. Not drawn in the opaque mode, where the faces already give the outline.
    if see and BOX_EDGE_LW > 0.0:
        (bx0, bx1), (bz0, bz1), (by0, by1) = geom["box_x"], geom["box_z"], geom["box_y"]
        for xa, xb, za, zb, ya, yb in (
                (bx0, bx1, bz0, bz0, by0, by0), (bx0, bx1, bz1, bz1, by0, by0),   #! bottom, along X
                (bx0, bx0, bz0, bz1, by0, by0), (bx1, bx1, bz0, bz1, by0, by0),   #! bottom, along Z
                (bx0, bx1, bz0, bz0, by1, by1), (bx0, bx1, bz1, bz1, by1, by1),   #! top, along X
                (bx0, bx0, bz0, bz1, by1, by1), (bx1, bx1, bz0, bz1, by1, by1),   #! top, along Z
                (bx0, bx0, bz0, bz0, by0, by1), (bx1, bx1, bz0, bz0, by0, by1),   #! verticals
                (bx0, bx0, bz1, bz1, by0, by1), (bx1, bx1, bz1, bz1, by0, by1)):
            ax.plot([xa, xb], [za, zb], [ya, yb], color=BOX_EDGE_COLOR,
                    linewidth=BOX_EDGE_LW, zorder=0)

    draw_ticks(ax, geom)
    setup_axes3d(ax, geom)
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    cb.ax.tick_params(labelsize=11)

#! REMOVED: draw_triad(). The orientation key is now the X / Z names drawn on each box by
#! draw_ticks(), so the separate triad axes -- and the TRIAD_H strip of white space it needed at
#! the bottom of every figure -- are gone. Nothing else referenced it.

def diag_figure(path, y, occ_frac, w_cat, y_lo, y_hi, w_med, nvalid, ncol, step_idx):
    """
    Side figure (--diag) that shows what the geometry measurement actually saw, so the numbers in
    the title can be checked instead of trusted:

      left   occupied fraction against Y, with the OCC_MIN cut and the resulting envelope. A clean
             sheet gives one plateau; a profile that never returns to zero means the threshold is
             sitting in the noise and the envelope will run away to the domain edge.
      right  distribution of the per-column thickness, with the median and the 16-84% band. A
             single narrow peak means Delta_CS is a real measurement; a broad or multi-peaked
             distribution means it is an average over structures of different size.
    """
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.0, 4.2))

    a1.plot(occ_frac, y, lw=1.2, color="#1f3b73")
    a1.axvline(OCC_MIN, ls="--", lw=1.0, color="0.45", label=f"--occ-min = {OCC_MIN:g}")
    a1.axhspan(y_lo, y_hi, color="#1f6b3b", alpha=0.12, label="envelope")
    a1.set_xscale("log"); a1.set_xlim(max(occ_frac[occ_frac > 0].min() * 0.5, 1.0e-6), 1.2)
    a1.set_ylim(float(y[0]), float(y[-1]))
    a1.set_xlabel("occupied fraction of columns"); a1.set_ylabel("Y")
    a1.set_title(f"occupancy  (envelope {y_lo:.2f} - {y_hi:.2f})", fontsize=11)
    a1.legend(fontsize=8, loc="upper left")

    if w_cat.size:
        a2.hist(w_cat, bins=60, color="#9c1f1f", alpha=0.75)
        a2.axvline(w_med, color="k", lw=1.4, label=f"median = {w_med:.2f}")
        a2.axvline(np.percentile(w_cat, 16.0), color="k", ls="--", lw=1.0)
        a2.axvline(np.percentile(w_cat, 84.0), color="k", ls="--", lw=1.0, label="16-84%")
        a2.legend(fontsize=8)
    a2.set_xlabel(r"per-column thickness  [$c/\omega_p$]"); a2.set_ylabel("columns")
    a2.set_title(f"thickness distribution  ({nvalid} of {ncol} columns used)", fontsize=11)

    fig.suptitle(f"step {step_idx:09d}  --  geometry diagnostics", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print(f"    diag -> {path}", flush=True)

#! ============================================================
#! Loop over the files assigned to this rank
#! ============================================================
for fname in files_local:
    step_idx = step_from_fname(fname)

    with Stream(fname, "r") as s:
        next(s.steps())
        #! coordinate arrays are 1D and tiny, so a full read is fine and gives the dimensions
        x = np.asarray(s.read("X1"), dtype=float); Nx = x.size
        y = np.asarray(s.read("X2"), dtype=float); Ny = y.size
        try:
            z = np.asarray(s.read("X3"), dtype=float); Nz = z.size
        except Exception:
            z, Nz = None, 1
        if z is None or Nz < 2:
            raise SystemExit(f"{fname} is not a 3D dump (Nz = {Nz}); this script renders 3D runs only")
        t_code = read_time(s)

        #! ---- sheet geometry, column by column on prof_nz Z slabs ----
        #! Every (x, z) column contributes its own crossings, so the kink enters the EXTENT (which
        #! is what the box needs) and never the THICKNESS (which is what the label quotes).
        wname = F_N if args.wvar == "rho" else F_JZ
        ks    = np.unique(np.linspace(0, Nz - 1, args.prof_nz).round().astype(int))
        occ_tot = np.zeros(Ny, dtype=np.int64)
        ncol_tot, nvalid_tot, w_all = 0, 0, []
        for k in ks:
            P = slab_xy_full(s, wname, k, Ny, Nx)
            P = np.abs(P) if args.wvar == "jz" else P
            wid, occ, ncol, nvalid = column_stats(P, y, args.efrac, args.wfrac, args.amp_min,
                                                  args.smooth, args.wref)
            occ_tot += occ; ncol_tot += ncol; nvalid_tot += nvalid
            w_all.append(wid[np.isfinite(wid)])

        w_cat    = np.concatenate(w_all) if w_all else np.array([])
        occ_frac = occ_tot / max(ncol_tot, 1)              #! fraction of columns occupied at each Y

        #! ---- envelope from the OCCUPANCY profile ----
        #! The sheet is "present" at a Y where at least OCC_MIN of all columns are above their own
        #! threshold. Shifting this boundary takes a whole population of columns, so particle noise
        #! and a few stray filaments cannot drag the box out to the domain edge the way taking each
        #! column's outermost crossing did.
        occupied = occ_frac >= OCC_MIN
        if not occupied.any():
            warn_once(f"noocc_{step_idx}",
                      f"step {step_idx}: no Y reaches the {OCC_MIN:g} occupied fraction in "
                      f"{args.wvar}; falling back to the full Y range (lower --occ-min)")
            y_ext_lo, y_ext_hi = float(y[0]), float(y[-1])
        else:
            jo = np.flatnonzero(occupied)
            y_ext_lo, y_ext_hi = float(y[jo[0]]), float(y[jo[-1]])

        if w_cat.size:
            w_meas = float(np.median(w_cat))
            #! 16th-84th percentile of the per-column thickness: a "1 sigma"-like spread. A band
            #! much wider than w_meas means the sheet is not uniformly thick (plasmoids, tearing),
            #! so the single quoted number is standing in for a varied structure -- and a band that
            #! spans an order of magnitude means the measurement itself is not to be trusted.
            w_lo16 = float(np.percentile(w_cat, 16.0))
            w_hi84 = float(np.percentile(w_cat, 84.0))
        else:
            warn_once(f"nowidth_{step_idx}",
                      f"step {step_idx}: no column in {args.wvar} passed the --amp-min test; the "
                      f"quoted thickness falls back to the envelope height")
            w_meas = y_ext_hi - y_ext_lo
            w_lo16 = w_hi84 = float("nan")

        #! Boxcar smoothing broadens the profile it is measuring. That is negligible while the
        #! smoothing length is a small fraction of the sheet, but on a sheet only a few cells thick
        #! it inflates the width by tens of percent -- so check it against the answer, not against
        #! an assumption about the grid.
        dy_cell = float(np.median(np.diff(y)))
        if args.smooth > 1 and np.isfinite(w_meas) and w_meas < 4.0 * args.smooth * dy_cell:
            warn_once(f"oversmooth_{step_idx}",
                      f"step {step_idx}: --smooth {args.smooth} spans {args.smooth * dy_cell:.2f} "
                      f"in Y but the sheet is only {w_meas:.2f} thick, so the quoted thickness is "
                      f"broadened by the smoothing; drop --smooth to about "
                      f"{max(int(w_meas / (4.0 * dy_cell)), 1)}")

        #! A spread wider than the median itself means the per-column widths disagree with each
        #! other by more than the number being quoted: say so rather than printing it as a fact.
        if np.isfinite(w_lo16) and w_meas > 0.0 and (w_hi84 - w_lo16) > w_meas:
            warn_once(f"wscatter_{step_idx}",
                      f"step {step_idx}: per-column thickness spread ({w_lo16:.2f}-{w_hi84:.2f}) is "
                      f"wider than its own median ({w_meas:.2f}); treat Delta_CS as indicative only "
                      f"-- run with --diag to see the distribution")

        y_c   = 0.5 * (y_ext_lo + y_ext_hi)                #! centre of the ENVELOPE, not of a profile
        y_exc = y_ext_hi - y_ext_lo                        #! full Y excursion of the sheet

        if args.diag:
            diag_figure(f"{OUTDIR}/CS_3D_diag_{step_idx:09d}.png", y, occ_frac, w_cat,
                        y_ext_lo, y_ext_hi, w_meas, nvalid_tot, ncol_tot, step_idx)

        #! ---- drawn Y window ----
        #! YWIN "extent" (default): the window is the measured ENVELOPE plus a margin of
        #!   wpad * thickness on each side. Because the envelope already contains every column's
        #!   crossings, the whole wiggling sheet is inside the box however large the kink grows --
        #!   the box simply gets taller as the instability develops, instead of clipping a half.
        #! YWIN "width": the old rule, y_c +- wpad * thickness. Kept for comparison; it clips once
        #!   the kink amplitude passes about wpad thicknesses.
        #! mode "centre" then stops the box at y_c (the envelope centre) in either rule.
        #! --ylim pins the window, so every frame spans the same Y and the projected box -- and
        #! therefore the image size -- is identical throughout a sequence. It is clamped to the
        #! domain, and if the measured sheet does not fit, that is reported rather than hidden:
        #! a fixed window CAN clip the sheet, which is the very failure the envelope logic exists
        #! to avoid, so it has to be a visible trade rather than a silent one.
        if YLIM is not None:
            if YLIM == "full":
                y_bot, y_top = float(y[0]), float(y[-1])
            else:
                y_bot = max(float(y[0]),  YLIM[0])
                y_top = min(float(y[-1]), YLIM[1])
            if y_ext_lo < y_bot or y_ext_hi > y_top:
                warn_once(f"ylimclip_{step_idx}",
                          f"step {step_idx}: --ylim [{y_bot:.2f}, {y_top:.2f}] does not contain the "
                          f"measured sheet [{y_ext_lo:.2f}, {y_ext_hi:.2f}]; the drawing is clipped "
                          f"-- widen --ylim or drop it to let the window follow the sheet")
        elif YWIN == "extent":
            margin = max(w_meas, 1.0e-30) * args.wpad
            y_bot  = max(float(y[0]),  y_ext_lo - margin)
            y_top  = min(float(y[-1]), y_ext_hi + margin)
        else:
            half   = max(w_meas, 1.0e-30) * args.wpad
            y_bot  = max(float(y[0]),  y_c - half)
            y_top  = min(float(y[-1]), y_c + half)
        if MODE == "centre" and YLIM is None:
            y_top = y_c
        if y_top <= y_bot:                                 #! degenerate (y_c sitting on the lower edge)
            y_top = min(float(y[-1]), y_bot + max(w_meas, 1.0e-30))

        #! Warn when the sheet runs into the Y boundary: the box is then clipped by the DOMAIN, not
        #! by this choice of window, and no setting here can show more of it.
        dy_tol = 1.5 * float(np.median(np.diff(y)))
        if y_ext_lo <= float(y[0]) + dy_tol or y_ext_hi >= float(y[-1]) - dy_tol:
            warn_once(f"ybound_{step_idx}",
                      f"step {step_idx}: the sheet reaches the Y boundary of the domain "
                      f"(envelope [{y_ext_lo:.2f}, {y_ext_hi:.2f}] in [{y[0]:.2f}, {y[-1]:.2f}]); "
                      f"the drawn box is clipped by the domain itself")
        j0 = int(np.searchsorted(y, y_bot, side="left"))
        j1 = int(np.searchsorted(y, y_top, side="right"))
        j0 = max(0, min(j0, Ny - 2)); j1 = max(j0 + 2, min(j1, Ny))
        nj = j1 - j0
        jt = j1 - 1                                        #! top face = last Y cell of the window
        jb = j0                                            #! bottom face = first Y cell of the window

        #! ---- the two walls that face the camera, so top + walls read as one solid box ----
        #! matplotlib puts the camera at azimuth `azim` in the (mpl-x, mpl-y) = (X, Z) plane, so the
        #! near edges are x[-1]/z[-1] when cos/sin of the azimuth are positive, and x[0]/z[0] otherwise
        az     = np.radians(args.azim)
        i_wall = Nx - 1 if np.cos(az) >= 0.0 else 0
        k_wall = Nz - 1 if np.sin(az) >= 0.0 else 0
        want_bottom = (args.elev < 0.0)                    #! only drawn when the camera is underneath

        #! ---- how many XZ planes, and where ----
        #! The stack is a SAMPLING of the volume in Y, so it obeys the usual sampling rule: a
        #! structure whose Y extent is comparable to the plane spacing is rendered as a handful of
        #! disconnected horizontal stripes rather than as the object it is. A flux rope lying along
        #! Z is the case that gives this away -- each plane cuts it in a long chord, so the rope
        #! reads as a flat ribbon in the stack while the SAME rope shows its circular cross-section
        #! on the near Z wall, which is drawn at full resolution. Circles on the wall with no
        #! cylinder behind them is the signature of too few slices, not of physics.
        #! "auto" therefore sizes the stack from the measured thickness rather than a fixed count.
        if SLICES == "auto":
            nsl = int(np.ceil((y_top - y_bot) / max(w_meas, 1.0e-30) * args.slices_per_w))
            nsl = int(np.clip(nsl, 5, args.slices_max))
        else:
            nsl = SLICES
        js = (np.unique(np.linspace(j0, j1 - 1, nsl).round().astype(int))
              if nsl > 1 else np.array([jt]))

        #! Spacing against the thickness -- if the ceiling bound, say so rather than quietly
        #! returning an aliased picture.
        d_sl = (y_top - y_bot) / max(len(js), 1)
        if len(js) > 1 and np.isfinite(w_meas) and d_sl > w_meas / 2.0:
            warn_once(f"alias_{step_idx}",
                      f"step {step_idx}: {len(js)} slices give {d_sl:.2f} spacing against a "
                      f"{w_meas:.2f} thick sheet, so Y structure is undersampled and tubes will "
                      f"read as flat stripes; raise --slices / --slices-max "
                      f"(about {int(np.ceil((y_top - y_bot) / w_meas * args.slices_per_w))} needed)")

        #! ---- the faces, for both fields ----
        F = {}
        for kind, name in (("jz", F_JZ), ("rho", F_N)):
            F[kind] = {"planes": [face_xz(s, name, int(j), Nz, Nx) for j in js],
                       "xw":  face_zy(s, name, i_wall, Nz, j0, nj),
                       "zw":  face_xy(s, name, k_wall, Nx, j0, nj),
                       "bot": face_xz(s, name, jb, Nz, Nx) if want_bottom else None}

    #! ---- per-axis spans; no assumption that Lx = Ly = Lz ----
    Lx, Ly, Lz = axis_span(x, "x1"), axis_span(y, "x2"), axis_span(z, "x3")

    #! ---- time in light-crossing times of the outflow direction ----
    time_label = (rf"$t\,c/L_x = {t_code / Lx:.2f}$" if np.isfinite(t_code) and Lx > 0
                  else f"step {step_idx:09d}")

    #! ---- block-average every face ----
    #! The walls get the full MAX_CELLS budget; the stacked planes get the coarser SLICE_CELLS one,
    #! because their cost is multiplied by the number of planes.
    sx, sy_, sz = stride_for(Nx), stride_for(nj), stride_for(Nz)
    cap = SLICE_CELLS if len(js) > 1 else MAX_CELLS         #! a single lid keeps the full detail
    px, pz = stride_for(Nx, cap), stride_for(Nz, cap)
    for kind in F:
        f = F[kind]
        F[kind] = {"planes": [block_mean(p, pz, px) for p in f["planes"]],   #! (Nz'', Nx'')
                   "xw":  block_mean(f["xw"], sy_, sz),                      #! (nj', Nz')
                   "zw":  block_mean(f["zw"], sy_, sx),                      #! (nj', Nx')
                   "bot": block_mean(f["bot"], pz, px) if f["bot"] is not None else None}

    #! ---- CELL-EDGE coordinates of the reduced grids ----
    #! One more point than cells, running from the true outer edge of the domain to the other, so
    #! the top face covers the FULL XZ plane and the walls meet it exactly with no seam and no
    #! missing strip at the far X / far Z side. Uniform spacing is assumed (Minkowski, uniform
    #! grid); with a short trailing block the last quad is up to one block too wide -- harmless
    #! here, but worth knowing if you ever run a non-uniform mesh.
    x0e, x1e = domain_edges(x)
    z0e, z1e = domain_edges(z)
    dy_c     = float(np.median(np.diff(y)))
    y0e      = float(y[j0]) - 0.5 * dy_c                    #! bottom edge of the first cell in the window
    y1e      = float(y[j1 - 1]) + 0.5 * dy_c                #! top edge of the last cell in the window

    xe = np.linspace(x0e, x1e, n_blocks(Nx, sx) + 1)         #! wall grids (MAX_CELLS budget)
    ze = np.linspace(z0e, z1e, n_blocks(Nz, sz) + 1)
    ye = np.linspace(y0e, y1e, n_blocks(nj, sy_) + 1)
    xp = np.linspace(x0e, x1e, n_blocks(Nx, px) + 1)         #! plane grids (SLICE_CELLS budget)
    zp = np.linspace(z0e, z1e, n_blocks(Nz, pz) + 1)

    #! ---- edge grids. mpl-x = X, mpl-y = Z, mpl-z = Y ----
    PX, PZ    = np.meshgrid(xp, zp)              #! XZ planes       : (Nz''+1, Nx''+1)
    WZ, WY_zy = np.meshgrid(ze, ye)              #! X wall          : (nj'+1, Nz'+1)
    WX, WY_xy = np.meshgrid(xe, ye)              #! Z wall          : (nj'+1, Nx'+1)

    #! Y positions of the drawn planes (cell centres of the sampled Y indices)
    plane_y = [float(y[int(j)]) for j in js]

    #! ---- box aspect ----
    #! Y_ASPECT is None ("true"): keep the PHYSICAL ratio Lx : Lz : (y1e - y0e), so a run with
    #! Lx = 2 Lz renders as a 2:1 slab whose thickness is the actual CS window. A number instead
    #! exaggerates Y to that fraction of L_x (vertical scale then not comparable to horizontal).
    H     = y1e - y0e
    y_box = H if Y_ASPECT is None else Y_ASPECT * Lx
    #! Drawn Z extent. XZ_ASPECT None keeps the physical Lz; a number R draws the box at X:Z = R:1
    #! regardless of the real Lz, which is how a thin guide direction is made visible. Only the
    #! box ASPECT changes -- the Z axis limits, the data and every measured quantity are untouched,
    #! so nothing is distorted except the on-screen proportions.
    z_box = Lz if XZ_ASPECT is None else Lx / XZ_ASPECT

    #! ---- axis limits with an equal fractional margin on every axis ----
    #! The same fraction on all three axes leaves the ratio of the ranges untouched, so the
    #! set_box_aspect proportions (Lx : Lz : y_box) still hold; the drawn box simply no longer
    #! touches the panel edge, which is what cut the left-top / right-bottom corners off.
    #! The X and Z ranges must also clear the tick marks, their labels AND the axis names, which
    #! sit outside the box by TICK_PAD and AXIS_LABEL_PAD respectively; otherwise the end labels or
    #! the "X" / "Z" are clipped at the panel edge. This is the only place the new labels cost
    #! anything, and it is a few per cent of the drawn width.
    pad_h = max(PAD_FRAC, max(TICK_PAD, AXIS_LABEL_PAD) + 0.05)

    def padded(lo, hi, pad=None):
        half = 0.5 * (hi - lo)
        mid  = 0.5 * (hi + lo)
        half = max(half, 1.0e-30) * (1.0 + (PAD_FRAC if pad is None else pad))
        return (mid - half, mid + half)

    geom  = {"PX": PX, "PZ": PZ, "WZ": WZ, "WY_zy": WY_zy, "WX": WX, "WY_xy": WY_xy,
             "plane_y": plane_y, "y_top": y1e, "y_bot": y0e,
             "box_x": (float(xe[0]), float(xe[-1])),            #! true box edges, for the wireframe
             "box_z": (float(ze[0]), float(ze[-1])),
             "box_y": (y0e, y1e),
             "x_wall": float(xe[-1] if i_wall else xe[0]),      #! on the edge axis, so the two
             "z_wall": float(ze[-1] if k_wall else ze[0]),      #! walls meet the top face exactly
             "xlim": padded(float(xe[0]), float(xe[-1]), pad_h),
             "zlim": padded(float(ze[0]), float(ze[-1]), pad_h),
             "ylim": padded(y0e, y1e), "box": (Lx, z_box, y_box)}

    #! ---- figure: the two field panels STACKED (J_z above, rho below) ----
    #! Stacking gives each panel the full figure width, which is what a wide flat slab needs; side
    #! by side, half of each panel was empty and the box had to be over-zoomed (and clipped) to
    #! fill it. The panel HEIGHT is taken from the projected shape of the box, so the figure is as
    #! tall as the picture actually is, and the box is then fitted into its slot exactly.
    #! panel height: from the projected shape unless --panel-h pins it. Pin it when the frames are
    #! going into a movie -- the sheet thickens with time, so an automatic height changes the image
    #! size from frame to frame and ffmpeg will refuse the sequence.
    #! The panel height is the PROJECTED height of the box at the drawn X width -- no more, so a
    #! flat sheet does not sit in a tall band of white space, and no less, so a thick one is not
    #! squeezed. The clip is only a sanity rail against a degenerate window; the previous 1.10
    #! ceiling was tight enough to bind on a tall box, and binding there is what used to shrink X.
    aspect  = ink_aspect(geom)                                  #! projected height / width
    draw_w  = PANEL_W - 2.0 * MARGIN_IN
    panel_h = (args.panel_h if args.panel_h is not None
               else float(np.clip(draw_w * aspect, 0.12 * draw_w, 2.00 * draw_w)))
    #! a pinned height was not chosen by the projection, so fall back to the containing fit there
    bind    = "fit" if args.panel_h is not None else "width"

    fig_w = PANEL_W + CB_IN                                     #! colour-bar strip on the right
    #! CHANGED: no triad strip -- just a thin BOTTOM_IN margin under the lower panel.
    fig_h = BOTTOM_IN + 2.0 * (panel_h + LABEL_IN) + TITLE_IN
    fig   = plt.figure(figsize=(fig_w, fig_h))

    def slot(y0_in):                                            #! inches (from the bottom) -> Bbox
        return Bbox.from_bounds(MARGIN_IN / fig_w, y0_in / fig_h,
                                (PANEL_W - 2.0 * MARGIN_IN) / fig_w, panel_h / fig_h)

    slot_dn = slot(BOTTOM_IN)                                   #! lower panel: rho
    slot_up = slot(BOTTOM_IN + panel_h + LABEL_IN)              #! upper panel: J_z

    panels = []
    for sl, kind, lbl in ((slot_up, "jz", r"$J_z$"), (slot_dn, "rho", r"$\rho$")):
        ax  = fig.add_axes(fitted_rect((fig_w, fig_h), sl, geom, bind), projection="3d")
        #! colour bar in the reserved right-hand strip, so it never shrinks the fitted panel
        cax = fig.add_axes([(PANEL_W + 0.16) / fig_w, sl.y0 + 0.22 * sl.height,
                            0.26 / fig_w,            0.56 * sl.height])
        render(fig, ax, kind, F[kind], geom, cax)
        fig.text(sl.x0 + 0.5 * sl.width, sl.y1 + 0.10 * LABEL_IN / fig_h, lbl,
                 fontsize=16, ha="center", va="bottom")
        panels.append(ax)

    #! Two separate numbers, because they mean different things once the sheet kinks: the median
    #! LOCAL thickness, and the full Y excursion of the sheet (thickness + kink amplitude).
    fig.text(0.5, 1.0 - 0.45 * TITLE_IN / fig_h, f"{time_label}", fontsize=20, ha="center", va="center")

    outfile = f"{OUTDIR}/CS_3D_{step_idx:09d}.png"
    #! no bbox_inches="tight": the layout above is already exact, and cropping to the artists would
    #! make the image size depend on the content (bad for a frame sequence)
    fig.savefig(outfile, dpi=DPI)
    plt.close(fig)
    plt.close("all")        #! probe figures + any axes orphaned by an exception path
    F = None 
    #! drawn vs native cell counts: if the drawn numbers are well below Nx / nj / Nz the faces are
    #! still being block-averaged and raising --max-cells will sharpen them further
    print(f"Saved {outfile}", flush=True)

comm.Barrier()
if rank == 0:
    print("\nDone.", flush=True)