"""
Created on Tue Oct 20 2026

@author: Pranab JD, Claude AI

3D rendering of the current and density of the colliding force-free flux tubes.

Geometry (Flux_Tubes_CG pgen)
-----------------------------
    The tubes are STRAIGHT CYLINDERS along Z, uniform in Z at t = 0; Z is periodic.

        mpl-x = X      the collision direction
        mpl-y = Y
        mpl-z = Z      the tube axis -> the cylinders stand upright in the figure

    The see-through stack is XY planes at a set of Z, i.e. a stack of tube CROSS-SECTIONS.

    --up picks which PHYSICAL axis is drawn vertically; only the drawing changes.
        z (default)  tube axis vertical, cylinders upright
        x            tube axis into the page, tubes STACKED vertically
        y            tube axis into the page, tubes side by side
    x and y also switch the default camera to elev 15 / azim -60.

Array axis order (Entity, 3D)
-----------------------------
    field arrays : (Nz, Ny, Nx)   [axis0 = x3, axis1 = x2, axis2 = x1]

Force-free diagnostic
---------------------
    The ff panel plots |J x B| / (|J| |B|) = sin(theta), which is zero for a perfectly
    force-free state. In 3D this needs a mask that the 2D script does not:

!   OUTSIDE the tubes B is purely z-directed and J is pure particle shot noise, so any cell
!   whose noise has J_z = 0 and J_x, J_y != 0 gives sin(theta) = 1 EXACTLY. Unmasked, the
!   surroundings would render as a solid opaque block of ones and hide the tubes completely
!   (the pgen's own CustomPostStep comment makes the same point about max-based residuals).
!   Cells are therefore drawn only where |J| clears a threshold, and everything below is
!   made fully transparent. The threshold is the LARGER of two, because the obvious one is
!   not enough on its own:
!     --ff-jmin  a fraction of the 99th percentile of |J|. On synthetic data with PIC-like
!                shot noise the noise floor sat at ~25% of that percentile, so the original
!                0.05 default kept 99.5% of the cells and the panel came out as a solid
!                block of ones. Default raised to 0.35 after measuring this.
!     --ff-nsig  multiples of the |J| noise floor MEASURED on the border ring of the middle
!                cross-section (outside both tubes by construction). This one tracks ppc
!                instead of being guessed at.
!   The ff panel is consequently a view of the TUBE INTERIORS only, and the threshold is
!   still a choice: the kept-cell fraction is printed for the first frame, and a warning
!   fires if it exceeds 50% (panel is noise) or falls below 1% (tubes masked away too).

Usage
-----
    folder="/scratch/project_465002528/pjd/flux_tubes/tubes/"
    fields="${folder}/fields"
    output="${folder}/plots_3d"

    srun python3 -u ../postprocessing/Plot_J_rho_3D.py "$fields" "$output" \
        --Lx 256.0 --Ly 256.0 --Lz 128.0 --r_j 64.0

    #! in-plane current instead of the default set
    srun python3 -u ../postprocessing/Plot_J_rho_3D.py "$fields" "$output" \
        --Lx 256.0 --Ly 256.0 --Lz 128.0 --fields Jx,Jy,Jz,absJ

    #! publication pass
    srun python3 -u ../postprocessing/Plot_J_rho_3D.py "$fields" "$output" \
        --Lx 256.0 --Ly 256.0 --Lz 128.0 --max-cells 720 --slices 24 --dpi 300

Flags worth knowing
-------------------
    --Lx --Ly --Lz   REQUIRED. Code-unit spans; checked against the file and the override wins.
    --r_j            optional; adds the t c/R label, as in the 2D scripts
    --fields         comma separated, from Jx, Jy, Jz, absJ, rho, ff. Default: Jz,absJ,rho,ff
    --ff-jmin        ff panel transparent below this fraction of the 99th-pct |J|. Default: 0.35
    --ff-nsig        ...or below this many times the measured |J| noise floor. Default: 3
    --slices         XY planes stacked through Z. Sets how SOLID the cylinders look, not
                     what is resolved (the tubes are uniform along Z). Cost is linear in
                     it. Default: 32; 8-12 is enough for a quick look.
    --alpha-gamma    raise to clear the background faster, lower to keep weak structure
    --elev --azim    camera. Defaults 30 / 225, the same as Plot_CS_3D.py
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
from mpl_toolkits.mplot3d import Axes3D, proj3d    # noqa: F401  (registers the 3d projection)

#! ============================================================
#! USER SETTINGS
#! ============================================================
TIME_KEY = "Time"


FIELDS = {
    "Jx":   dict(key="fJ1", cmap="seismic", lim=0.3,        alpha="centre",   label=r"$J_x$"),
    "Jy":   dict(key="fJ2", cmap="seismic", lim=0.3,        alpha="centre",   label=r"$J_y$"),
    "Jz":   dict(key="fJ3", cmap="seismic", lim=0.6,        alpha="centre",   label=r"$J_z$"),
    "absJ": dict(key=None,  cmap="inferno", lim=0.6,        alpha="low",      label=r"$|J|$"),
    "rho":  dict(key="fN",  cmap="inferno", lim=None,       alpha="baseline", floor=0.30,
                 label=r"$\rho$"),
    "ff":   dict(key=None,  cmap="inferno", lim=(0.0, 1.0), alpha="low",
                 label=r"$|J\times B|/(|J||B|)$"),
}
DEFAULT_FIELDS = "Jz,absJ,rho,ff"

PCT = 99.0

MAX_CELLS   = 480
SLICES      = 32

SLICE_CELLS = 200
ALPHA_MAX   = 0.85
ALPHA_GAMMA = 1.6
BG_FRAC     = 0.06
FF_JMIN     = 0.35       #! ff panel transparent below this fraction of the 99th-pct |J|
FF_NSIG     = 3.0        #! ...or below this many times the MEASURED |J| noise floor
BOX_EDGE_LW    = 0.6
BOX_EDGE_COLOR = "0.55"

PANEL_W   = 5.6
CB_IN     = 0.80
LABEL_IN  = 0.28
TITLE_IN  = 0.44
BOTTOM_IN = 0.10
MARGIN_IN = 0.06
COL_GAP   = 0.10

TICK_N     = 4
TICK_FS    = 10
TICK_LEN   = 0.030
TICK_PAD   = 0.055
TICK_COLOR = "0.25"

AXIS_LABEL       = {"x": "X", "y": "Y"}
AXIS_LABEL_FS    = 14
AXIS_LABEL_PAD   = 0.200   #! must exceed TICK_PAD, or the name overprints a tick number
AXIS_LABEL_AT    = 0.62
AXIS_LABEL_COLOR = "0.10"

ZOOM     = 1.0
PAD_FRAC = 0.08
DPI      = 200

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

parser.add_argument("--Lx", type=float, required=True, help="Box length along X in code units")
parser.add_argument("--Ly", type=float, required=True, help="Box length along Y in code units")
parser.add_argument("--Lz", type=float, required=True, help="Box length along Z (tube axis) in code units")
parser.add_argument("--r_j", type=float, default=None,
                    help="tube radius in code units; adds the t c/R label, as in the 2D scripts")

parser.add_argument("--fields", type=str, default=DEFAULT_FIELDS,
                    help=f"comma separated panels from {', '.join(FIELDS)}. Default: {DEFAULT_FIELDS}")
parser.add_argument("--ff-jmin", type=float, default=FF_JMIN, dest="ff_jmin",
                    help=f"the ff panel is transparent where |J| is below this fraction of the "
                         f"99th percentile of |J|. Default: {FF_JMIN}")
parser.add_argument("--ff-nsig", type=float, default=FF_NSIG, dest="ff_nsig",
                    help=f"...or below this many times the |J| noise floor measured outside the "
                         f"tubes. The LARGER of the two thresholds is used, because a fixed "
                         f"fraction of the peak does not track the shot noise as ppc changes. "
                         f"Default: {FF_NSIG}")

parser.add_argument("--slices",      type=int,   default=SLICES,
                    help=f"XY planes stacked along Z. Default: {SLICES}")
parser.add_argument("--slice-cells", type=int,   default=SLICE_CELLS, dest="slice_cells")
parser.add_argument("--max-cells",   type=int,   default=MAX_CELLS,   dest="max_cells")
parser.add_argument("--alpha-max",   type=float, default=ALPHA_MAX,   dest="alpha_max")
parser.add_argument("--alpha-gamma", type=float, default=ALPHA_GAMMA, dest="alpha_gamma")
parser.add_argument("--bg-frac",     type=float, default=BG_FRAC,     dest="bg_frac")
parser.add_argument("--up", type=str, default="z", choices=["z", "y", "x"],
                    help="which PHYSICAL axis is drawn vertically. 'z' (default): the tube axis "
                         "is vertical, so the cylinders stand upright. 'x': the tube axis runs "
                         "into the page and the two tubes are STACKED vertically, the Granier et "
                         "al. (2025) arrangement, because this pgen separates them along X. 'y': "
                         "tubes lie along the depth axis but side by side. Only the drawing "
                         "changes -- the data, the axis names and every measured quantity are "
                         "untouched. 'x' and 'y' also switch the default camera to elev 15 / "
                         "azim -60 unless --elev / --azim are given.")
parser.add_argument("--elev", type=float, default=None)
parser.add_argument("--azim", type=float, default=None)
parser.add_argument("--dpi",  type=int,   default=DPI)
parser.add_argument("--zoom", type=float, default=ZOOM)
parser.add_argument("--pad-frac", type=float, default=PAD_FRAC, dest="pad_frac")
parser.add_argument("--panel-h",  type=float, default=None,     dest="panel_h")
args = parser.parse_args()

#! Which physical axis goes up, and the camera that suits it. Resolved BEFORE anything
#! else uses args.elev / args.azim.
UP = args.up
#! ORDER[i] is the PHYSICAL axis drawn on matplotlib axis i (x, y, z = right, depth, up).
#! The vertical slot is the one --up names; the other two keep their cyclic order.
ORDER = {"z": ("X", "Y", "Z"),
         "y": ("X", "Z", "Y"),
         "x": ("Y", "Z", "X")}[UP]
SLOT = {name: i for i, name in enumerate(ORDER)}    #! physical axis -> drawn axis index

if args.elev is None:
    args.elev = 30.0 if UP == "z" else 15.0
if args.azim is None:
    args.azim = 225.0 if UP == "z" else -60.0

MAX_CELLS, SLICE_CELLS = args.max_cells, args.slice_cells
ALPHA_MAX, ALPHA_GAMMA = args.alpha_max, args.alpha_gamma
DPI, ZOOM, PAD_FRAC    = args.dpi, args.zoom, args.pad_frac
BG_FRAC, FF_JMIN       = args.bg_frac, args.ff_jmin
FF_NSIG                = args.ff_nsig
SLICES                 = max(int(args.slices), 1)

PANELS = [f.strip() for f in args.fields.split(",") if f.strip()]
for f in PANELS:
    if f not in FIELDS:
        parser.error(f"unknown field '{f}'; choose from {', '.join(FIELDS)}")
if not PANELS:
    parser.error("--fields selected nothing")

for nm, v in (("--Lx", args.Lx), ("--Ly", args.Ly), ("--Lz", args.Lz)):
    if v <= 0.0:
        parser.error(f"{nm} must be positive")
if args.r_j is not None and args.r_j <= 0.0:
    parser.error("--r_j must be positive")
if MAX_CELLS < 16 or SLICE_CELLS < 16:
    parser.error("--max-cells and --slice-cells must be >= 16")
if not (0.0 < ALPHA_MAX <= 1.0):
    parser.error("--alpha-max must lie in (0, 1]")
if ALPHA_GAMMA <= 0.0:
    parser.error("--alpha-gamma must be positive")
if not (0.0 < BG_FRAC < 0.5):
    parser.error("--bg-frac must lie in (0, 0.5)")
if not (0.0 <= FF_JMIN < 1.0):
    parser.error("--ff-jmin must lie in [0, 1)")
if FF_NSIG < 0.0:
    parser.error("--ff-nsig must be >= 0")
if DPI < 50:
    parser.error("--dpi must be >= 50")
if ZOOM <= 0.0:
    parser.error("--zoom must be positive")
if PAD_FRAC < 0.0:
    parser.error("--pad-frac must be >= 0")
if args.panel_h is not None and args.panel_h <= 0.0:
    parser.error("--panel-h must be positive")

BASE, OUTDIR = args.base, args.outdir
L_OVER = {"x1": args.Lx, "x2": args.Ly, "x3": args.Lz}
L_FLAG = {"x1": "--Lx", "x2": "--Ly", "x3": "--Lz"}

NEED_B = "ff" in PANELS                       #! B is read only for the force-free residual
NEED_J = any(n in ("Jx", "Jy", "Jz", "absJ", "ff") for n in PANELS)

if rank == 0:
    os.makedirs(OUTDIR, exist_ok=True)
comm.Barrier()

files = sorted(glob.glob(f"{BASE}/fields.*.bp"))
if rank == 0:
    print(f"\nFound {len(files)} files", flush=True)
    print(f"    box      = {args.Lx} x {args.Ly} x {args.Lz}  [d_e]   (tube axis = Z)", flush=True)
    print(f"    panels   = {', '.join(PANELS)}", flush=True)
    print(f"    stack    = {SLICES} XY planes at {SLICE_CELLS} cells/side, "
          f"walls at {MAX_CELLS} cells/side", flush=True)
    print(f"    alpha    = max {ALPHA_MAX:g}, gamma {ALPHA_GAMMA:g}, background ring {BG_FRAC:g}", flush=True)
    if "ff" in PANELS:
        print(f"    ff panel = masked below max({FF_JMIN:g} x p99|J|, {FF_NSIG:g} x noise floor) "
              f"(outside the tubes the residual is pure noise and pins at 1)", flush=True)
    print(f"    view     = (elev {args.elev:g}, azim {args.azim:g}), dpi {DPI}\n", flush=True)
files_local = files[rank::size]

#! ============================================================
#! Helpers
#! ============================================================
_warned = set()

def warn_once(key, msg):
    if rank == 0 and key not in _warned:
        _warned.add(key)
        print(f"WARNING: {msg}", flush=True)

#! ---- the ONLY place the drawn orientation lives ----
def to_mpl(x_, y_, z_):
    """
    Physical (X, Y, Z) -> matplotlib (x, y, z). Works on scalars and on arrays alike,
    because it only reorders its arguments.

        --up z : (X, Y, Z)   tube axis vertical, cylinders upright
        --up y : (X, Z, Y)   tubes lie along the depth axis, side by side
        --up x : (Y, Z, X)   tubes lie along the depth axis, STACKED vertically

    Every coordinate handed to plot_surface, to the wireframe and to the tick artists goes
    through here, so adding an orientation never means editing the drawing code twice.
    """
    p = {"X": x_, "Y": y_, "Z": z_}
    return (p[ORDER[0]], p[ORDER[1]], p[ORDER[2]])

def near_index(axis, n):
    """
    Index of the face of `axis` that faces the camera, given where that physical axis is
    drawn. A face whose axis is drawn VERTICAL is horizontal on screen (a floor or a
    ceiling) and is chosen by the elevation; the other two are chosen by the azimuth,
    exactly as matplotlib places the camera in the (drawn x, drawn y) plane.
    """
    az = np.radians(args.azim)
    slot = SLOT[axis]
    if slot == 0:
        return n - 1 if np.cos(az) >= 0.0 else 0
    if slot == 1:
        return n - 1 if np.sin(az) >= 0.0 else 0
    return 0 if args.elev >= 0.0 else n - 1

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

def axis_span(coord, name):
    c = np.asarray(coord, dtype=float).ravel()
    if c.size < 2:
        return float(L_OVER[name])
    d  = np.diff(c)
    dc = float(np.median(d))
    if dc <= 0.0:
        warn_once(f"desc_{name}", f"{name} is not increasing; the extent will be inverted")
    eps = float(np.finfo(np.float32).eps) * float(np.max(np.abs(c)))
    if np.max(np.abs(d - dc)) > max(1.0e-6 * abs(dc), 8.0 * eps):
        warn_once(f"nonuni_{name}", f"{name} spacing is non-uniform; the span uses the median dx")
    span  = float((c[-1] + 0.5 * dc) - (c[0] - 0.5 * dc))
    L_cli = float(L_OVER[name])
    if span > 0.0 and abs(span - L_cli) / span > 0.01:
        warn_once(f"mismatch_{name}",
                  f"{L_FLAG[name]} = {L_cli:g} disagrees with the {name} span {span:g} from the "
                  f"file by more than 1%; using the override")
    return L_cli

def cell_width(L, n):
    """Exact cell width from the box length and the cell count; constant for the whole run."""
    return float(L) / float(n)

def domain_edges(c, L, n):
    """Outer cell EDGES of an axis, built from the exact dx rather than from the coordinates."""
    dx = cell_width(L, n)
    x0 = float(np.asarray(c, dtype=float).ravel()[0]) - 0.5 * dx
    return x0, x0 + float(L)

#! ---- hyperslab reads ----
def face_xy(stream, name, k, Ny, Nx):
    """Tube CROSS-SECTION at one Z index -> (Ny, Nx): rows = Y, columns = X."""
    a = np.asarray(stream.read(name, start=[int(k), 0, 0], count=[1, int(Ny), int(Nx)]))
    return a.reshape(int(Ny), int(Nx))

def face_yz(stream, name, i, Nz, Ny):
    """Vertical wall at one X index -> (Nz, Ny): rows = Z, columns = Y."""
    a = np.asarray(stream.read(name, start=[0, 0, int(i)], count=[int(Nz), int(Ny), 1]))
    return a.reshape(int(Nz), int(Ny))

def face_xz(stream, name, j, Nz, Nx):
    """Vertical wall at one Y index -> (Nz, Nx): rows = Z, columns = X."""
    a = np.asarray(stream.read(name, start=[0, int(j), 0], count=[int(Nz), 1, int(Nx)]))
    return a.reshape(int(Nz), int(Nx))

def border_median(a, frac):
    """
    Median over the outer `frac` ring of a 2D face, used as the background for the
    "baseline" alpha rule. Valid only while the tubes do not reach the domain border --
    true for this geometry (centres at Lx/2 -+ R with R ~ Lx/4) but not a general guarantee.
    """
    a = np.asarray(a, dtype=float)
    ny, nx = a.shape
    n = max(1, int(round(frac * min(ny, nx))))
    ring = np.concatenate([a[:n, :].ravel(), a[-n:, :].ravel(),
                           a[:, :n].ravel(), a[:, -n:].ravel()])
    ring = ring[np.isfinite(ring)]
    return float(np.median(ring)) if ring.size else 0.0

#! ---- downsampling ----
def stride_for(n, cap):
    return max(1, int(np.ceil(n / float(cap))))

def _reduce_mean(a, s, axis):
    if s == 1:
        return a
    n   = a.shape[axis]
    idx = np.arange(0, n, s)
    cnt = np.diff(np.append(idx, n)).astype(float)
    out = np.add.reduceat(a, idx, axis=axis)
    shp = [1] * out.ndim
    shp[axis] = cnt.size
    return out / cnt.reshape(shp)

def block_mean(a, sr, sc):
    return _reduce_mean(_reduce_mean(np.asarray(a, dtype=float), sr, 0), sc, 1)

def n_blocks(n, s):
    return int(np.ceil(n / float(s)))

#! ---- the force-free residual ----
def ff_residual(J, B):
    """
    sin(theta) between J and B, cell by cell. J and B are 3-tuples of same-shaped arrays.
    Cells with |J| = 0 or |B| = 0 return 0 rather than a NaN; they are masked out anyway.
    """
    (Jx, Jy, Jz), (Bx, By, Bz) = J, B
    rx, ry, rz = Jy * Bz - Jz * By, Jz * Bx - Jx * Bz, Jx * By - Jy * Bx
    num = np.sqrt(rx ** 2 + ry ** 2 + rz ** 2)
    den = np.sqrt(Jx ** 2 + Jy ** 2 + Jz ** 2) * np.sqrt(Bx ** 2 + By ** 2 + Bz ** 2)
    out = np.zeros_like(num)
    m   = den > 0.0
    out[m] = num[m] / den[m]
    return out

#! ---- rendering ----
def colour_limits(spec, faces):
    allv = np.concatenate([f.ravel() for f in faces if f is not None])
    allv = allv[np.isfinite(allv)]
    if allv.size == 0:
        return 0.0, 1.0
    lim, diverging = spec["lim"], (spec["cmap"] == "seismic")
    if lim is None:
        if diverging:
            v = max(float(np.percentile(np.abs(allv), PCT)), 1.0e-30)
            return -v, v
        lo = float(np.percentile(allv, 100.0 - PCT))
        hi = float(np.percentile(allv, PCT))
        return lo, max(hi, lo + 1.0e-30)
    if np.isscalar(lim):
        return (-float(lim), float(lim)) if diverging else (0.0, float(lim))
    return float(lim[0]), float(lim[1])

def face_rgba(cmap, norm, data, mode, base, vmin, vmax, see_through, mask=None, floor=0.0):
    """
    Per-cell RGBA: the COLOUR is the usual normalised map, the ALPHA says how far the cell is
    from this field's background. `mask` (False -> fully transparent) is how the ff panel
    drops the cells where there is no current to speak of. `floor` is the minimum alpha
    denominator for the baseline rule (see FIELDS).
    """
    t    = np.asarray(norm(data), dtype=float)
    rgba = cmap(t)
    if not see_through and mask is None:
        return rgba

    if mode == "centre":
        m = np.abs(2.0 * t - 1.0)
    elif mode == "baseline":
        #! `floor` keeps the denominator off the noise scale when the colour window is
        #! percentile-scaled and the field has no real structure.
        denom = max(abs(vmax - base), abs(vmin - base), float(floor), 1.0e-30)
        m = np.abs(np.asarray(data, dtype=float) - base) / denom
    else:
        m = t

    a = np.clip(np.nan_to_num(m), 0.0, 1.0) ** ALPHA_GAMMA * ALPHA_MAX
    if not see_through:
        a = np.ones_like(a)
    if mask is not None:
        a = np.where(mask, a, 0.0)
    rgba[..., 3] = a
    return rgba

def surf(ax, MX, MY, MZ, rgba):
    ax.plot_surface(MX, MY, MZ, facecolors=rgba, rstride=1, cstride=1,
                    shade=False, linewidth=0.0, edgecolor="none", antialiased=True)

def setup_axes3d(ax, geom):
    ax.set_xlim(geom["xlim"])        #! mpl-x = X
    ax.set_ylim(geom["ylim"])        #! mpl-y = Y
    ax.set_zlim(geom["zlim"])        #! mpl-z = Z, the tube axis
    try:
        ax.set_box_aspect(geom["box"], zoom=ZOOM)
    except TypeError:
        ax.set_box_aspect(geom["box"])
        warn_once("nozoom", "this matplotlib has no zoom argument for set_box_aspect; the "
                            "panels will carry extra white space")
    ax.view_init(elev=args.elev, azim=args.azim)
    ax.set_axis_off()
    ax.patch.set_visible(False)

def _ink_bbox(fig, ax, geom):
    M   = ax.get_proj()
    pts = np.array([proj3d.proj_transform(a, b, c, M)[:2]
                    for a in geom["xlim"] for b in geom["ylim"] for c in geom["zlim"]])
    f = fig.transFigure.inverted().transform(ax.transData.transform(pts))
    return Bbox.from_extents(f[:, 0].min(), f[:, 1].min(), f[:, 0].max(), f[:, 1].max())

def probe_ink(figsize, rect, geom):
    f = plt.figure(figsize=figsize)
    a = f.add_axes(rect, projection="3d")
    setup_axes3d(a, geom)
    f.canvas.draw()
    bb = _ink_bbox(f, a, geom)
    plt.close(f)
    return bb

_ink_cache = {}

def ink_aspect(geom):
    key = (geom["xlim"], geom["ylim"], geom["zlim"], args.elev, args.azim)
    if key not in _ink_cache:
        probe = probe_ink((10.0, 10.0), [0.05, 0.05, 0.90, 0.90], geom)
        _ink_cache[key] = float(probe.height / max(probe.width, 1.0e-12))
    return _ink_cache[key]

def fitted_rect(figsize, slot, geom, bind="width"):
    ink = probe_ink(figsize, [slot.x0, slot.y0, slot.width, slot.height], geom)
    sw  = slot.width / max(ink.width, 1.0e-12)
    s   = sw if bind == "width" else min(sw, slot.height / max(ink.height, 1.0e-12))
    nw, nh = slot.width * s, slot.height * s
    rx = (ink.x0 + 0.5 * ink.width  - slot.x0) / slot.width
    ry = (ink.y0 + 0.5 * ink.height - slot.y0) / slot.height
    return [slot.x0 + 0.5 * slot.width  - rx * nw,
            slot.y0 + 0.5 * slot.height - ry * nh, nw, nh]

def draw_ticks(ax, geom):
    #! Everything here is in MPL space, so the two edges drawn are always "the two bottom
    #! edges facing the camera" whatever --up is; only their physical NAMES change.
    (x0, x1), (y0, y1) = geom["box_mx"], geom["box_my"]
    z0 = geom["box_mz"][0]
    dx, dy = x1 - x0, y1 - y0
    outy = 1.0 if geom["wall_my"] >= 0.5 * (y0 + y1) else -1.0
    outx = 1.0 if geom["wall_mx"] >= 0.5 * (x0 + x1) else -1.0
    fx = "%.0f" if dx >= 10.0 else "%.2g"
    fy = "%.0f" if dy >= 10.0 else "%.2g"

    for f in np.linspace(0.0, 1.0, TICK_N + 1):
        xv = x0 + f * dx
        ax.plot([xv, xv], [geom["wall_my"], geom["wall_my"] + outy * TICK_LEN * dy], [z0, z0],
                color=TICK_COLOR, lw=0.9, zorder=5)
        ax.text(xv, geom["wall_my"] + outy * TICK_PAD * dy, z0, fx % xv,
                color=TICK_COLOR, fontsize=TICK_FS, ha="center", va="center", zorder=5)

    for f in np.linspace(0.0, 1.0, TICK_N + 1):
        yv = y0 + f * dy
        ax.plot([geom["wall_mx"], geom["wall_mx"] + outx * TICK_LEN * dx], [yv, yv], [z0, z0],
                color=TICK_COLOR, lw=0.9, zorder=5)
        ax.text(geom["wall_mx"] + outx * TICK_PAD * dx, yv, z0, fy % yv,
                color=TICK_COLOR, fontsize=TICK_FS, ha="center", va="center", zorder=5)

    ax.text(x0 + AXIS_LABEL_AT * dx, geom["wall_my"] + outy * AXIS_LABEL_PAD * dy, z0, geom["name_mx"],
            color=AXIS_LABEL_COLOR, fontsize=AXIS_LABEL_FS, fontweight="bold",
            ha="center", va="center", zorder=6)
    ax.text(geom["wall_mx"] + outx * AXIS_LABEL_PAD * dx, y0 + AXIS_LABEL_AT * dy, z0, geom["name_my"],
            color=AXIS_LABEL_COLOR, fontsize=AXIS_LABEL_FS, fontweight="bold",
            ha="center", va="center", zorder=6)

def render(fig, ax, name, F, geom, cax):
    spec = FIELDS[name]
    cmap = plt.get_cmap(spec["cmap"])
    vmin, vmax = colour_limits(spec, F["planes"] + [F["xw"], F["yw"]])
    norm = Normalize(vmin, vmax)
    mode, base = spec["alpha"], F["base"]
    floor = spec.get("floor", 0.0)
    see  = len(F["planes"]) > 1

    #! Each surf call is given PHYSICAL (X, Y, Z) grids and to_mpl puts them on the drawn
    #! axes, so the same three calls serve every orientation.
    mk = F.get("mask", {})
    for z_val, dat, msk in zip(geom["plane_z"], F["planes"],
                               mk.get("planes", [None] * len(F["planes"]))):
        surf(ax, *to_mpl(geom["PX"], geom["PY"], np.full_like(geom["PX"], z_val)),
             face_rgba(cmap, norm, dat, mode, base, vmin, vmax, see, msk, floor))
    surf(ax, *to_mpl(np.full_like(geom["WY"], geom["x_wall"]), geom["WY"], geom["WZ_yz"]),
         face_rgba(cmap, norm, F["xw"], mode, base, vmin, vmax, see, mk.get("xw"), floor))
    surf(ax, *to_mpl(geom["WX"], np.full_like(geom["WX"], geom["y_wall"]), geom["WZ_xz"]),
         face_rgba(cmap, norm, F["yw"], mode, base, vmin, vmax, see, mk.get("yw"), floor))

    if see and BOX_EDGE_LW > 0.0:
        (bx0, bx1), (by0, by1), (bz0, bz1) = geom["box_x"], geom["box_y"], geom["box_z"]
        for xa, xb, ya, yb, za, zb in (
                (bx0, bx1, by0, by0, bz0, bz0), (bx0, bx1, by1, by1, bz0, bz0),
                (bx0, bx0, by0, by1, bz0, bz0), (bx1, bx1, by0, by1, bz0, bz0),
                (bx0, bx1, by0, by0, bz1, bz1), (bx0, bx1, by1, by1, bz1, bz1),
                (bx0, bx0, by0, by1, bz1, bz1), (bx1, bx1, by0, by1, bz1, bz1),
                (bx0, bx0, by0, by0, bz0, bz1), (bx1, bx1, by0, by0, bz0, bz1),
                (bx0, bx0, by1, by1, bz0, bz1), (bx1, bx1, by1, by1, bz0, bz1)):
            ax.plot(*to_mpl([xa, xb], [ya, yb], [za, zb]), color=BOX_EDGE_COLOR,
                    linewidth=BOX_EDGE_LW, zorder=0)

    draw_ticks(ax, geom)
    setup_axes3d(ax, geom)
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    cb.ax.tick_params(labelsize=14)

#! ============================================================
#! Loop over the files assigned to this rank
#! ============================================================
first_frame = True

for fname in files_local:
    step_idx = step_from_fname(fname)

    with Stream(fname, "r") as s:
        next(s.steps())

        x = np.asarray(s.read("X1"), dtype=float); Nx = x.size
        y = np.asarray(s.read("X2"), dtype=float); Ny = y.size
        try:
            z = np.asarray(s.read("X3"), dtype=float); Nz = z.size
        except Exception:
            z, Nz = None, 1
        if z is None or Nz < 2:
            raise SystemExit(f"{fname} is not a 3D dump (Nz = {Nz}); use Plot_J_rho.py for 2D runs")

        t_code = read_time(s)

        i_wall = near_index("X", Nx)
        j_wall = near_index("Y", Ny)
        k_near = near_index("Z", Nz)
        ks     = np.unique(np.linspace(0, Nz - 1, SLICES).round().astype(int))

        def read_all(key):
            return {"planes": [face_xy(s, key, int(k), Ny, Nx) for k in ks],
                    "xw":     face_yz(s, key, i_wall, Nz, Ny),
                    "yw":     face_xz(s, key, j_wall, Nz, Nx)}

        raw = {}
        if NEED_J:
            for comp, key in (("J1", "fJ1"), ("J2", "fJ2"), ("J3", "fJ3")):
                raw[comp] = read_all(key)
        if NEED_B:
            for comp, key in (("B1", "fB1"), ("B2", "fB2"), ("B3", "fB3")):
                raw[comp] = read_all(key)
        if "rho" in PANELS:
            raw["N"] = read_all("fN")

        any_comp = next(iter(raw))
        got = raw[any_comp]["planes"][0].shape
        if got != (Ny, Nx):
            raise SystemExit(f"{fname}: an XY plane came back as {got}, expected {(Ny, Nx)}. "
                             f"The file is not in the (Nz, Ny, Nx) order this script assumes.")

    #! ---- assemble the panels ----
    PARTS = ("planes", "xw", "yw")

    def per_part(fn):
        """Apply fn to each face of a panel; `planes` is a list, the walls are single arrays."""
        return {"planes": [fn("planes", i) for i in range(len(ks))],
                "xw": fn("xw", None), "yw": fn("yw", None)}

    def get(comp, part, i):
        return raw[comp][part][i] if part == "planes" else raw[comp][part]

    F = {}
    for name in PANELS:
        spec = FIELDS[name]
        if name in ("Jx", "Jy", "Jz"):
            comp = {"Jx": "J1", "Jy": "J2", "Jz": "J3"}[name]
            F[name] = {k: raw[comp][k] for k in PARTS}
        elif name == "rho":
            F[name] = {k: raw["N"][k] for k in PARTS}
        elif name == "absJ":
            F[name] = per_part(lambda p, i: np.sqrt(get("J1", p, i) ** 2 +
                                                    get("J2", p, i) ** 2 +
                                                    get("J3", p, i) ** 2))
        else:                                   #! "ff"
            F[name] = per_part(lambda p, i: ff_residual(
                (get("J1", p, i), get("J2", p, i), get("J3", p, i)),
                (get("B1", p, i), get("B2", p, i), get("B3", p, i))))

        mid = F[name]["planes"][len(F[name]["planes"]) // 2]
        F[name]["base"] = border_median(mid, BG_FRAC) if spec["alpha"] == "baseline" else 0.0

    #! ---- the ff mask ----
    #! Built at FULL resolution from |J|, then block-averaged alongside the data so the mask
    #! and the field it hides stay on the same grid. The threshold is set from the 99th
    #! percentile of |J| over the drawn faces of this frame, so it follows the current as the
    #! tubes evolve instead of being pinned to the t = 0 amplitude.
    if "ff" in PANELS:
        absJ = per_part(lambda p, i: np.sqrt(get("J1", p, i) ** 2 +
                                             get("J2", p, i) ** 2 +
                                             get("J3", p, i) ** 2))
        allJ = np.concatenate([a.ravel() for a in absJ["planes"]] +
                              [absJ["xw"].ravel(), absJ["yw"].ravel()])
        allJ = allJ[np.isfinite(allJ)]
        j_p99 = float(np.percentile(allJ, 99.0)) if allJ.size else 0.0

        #! TWO thresholds, and the LARGER wins.
        #!   --ff-jmin  a fraction of the peak current. Simple, but on its own it does not
        #!              work: the deposited |J| outside the tubes is shot noise whose floor
        #!              is a sizeable fraction of the peak (measured at ~25% of the 99th
        #!              percentile on synthetic data with PIC-like noise), so any fraction
        #!              below that keeps the whole box and the panel renders as a solid
        #!              block of ones -- the exact failure this mask exists to prevent.
        #!   --ff-nsig  multiples of the MEASURED noise floor, taken from the border ring of
        #!              the middle cross-section, which is outside both tubes and therefore
        #!              is the noise floor by construction. This one adapts to ppc instead
        #!              of being guessed.
        j_floor = border_median(absJ["planes"][len(absJ["planes"]) // 2], BG_FRAC)
        j_thr   = max(FF_JMIN * j_p99, FF_NSIG * j_floor)

        F["ff"]["mask_full"] = {"planes": [a > j_thr for a in absJ["planes"]],
                                "xw": absJ["xw"] > j_thr, "yw": absJ["yw"] > j_thr}
        if first_frame:
            frac = float(np.mean(allJ > j_thr)) if allJ.size else 0.0
            # print(f"[rank {rank}] step {step_idx:09d}  ff mask keeps {100.0 * frac:.1f}% of the "
            #       f"drawn cells  (|J| > {j_thr:.4g}; p99 {j_p99:.4g}, noise floor {j_floor:.4g})",
            #       flush=True)
            # #! If this keeps nearly everything the panel is noise, not physics; if it keeps
            # #! almost nothing the tubes have been masked away too. Say so rather than
            # #! letting a meaningless picture out silently.
            # if frac > 0.5:
            #     warn_once("ff_mask_loose",
            #               f"the ff mask keeps {100.0 * frac:.0f}% of the drawn cells, so the "
            #               f"panel is dominated by the shot-noise background (where the residual "
            #               f"pins at 1 and means nothing). Raise --ff-jmin or --ff-nsig.")
            # elif frac < 0.01:
            #     warn_once("ff_mask_tight",
            #               f"the ff mask keeps only {100.0 * frac:.2f}% of the drawn cells; the "
            #               f"tube interiors are probably being masked away too. Lower --ff-jmin "
            #               f"or --ff-nsig.")

    Lx, Ly, Lz = axis_span(x, "x1"), axis_span(y, "x2"), axis_span(z, "x3")

    if np.isfinite(t_code):
        time_label = rf"$t\,c/L_x = {t_code / Lx:.3f}$"
        if args.r_j is not None:
            time_label += rf"   |   $t\,c/R = {t_code / args.r_j:.3f}$"
    else:
        time_label = f"step {step_idx:09d}   (time not found - check TIME_KEY)"

    #! ---- block-average every face (and the mask, as a float then re-thresholded) ----
    sx, sy, sz = stride_for(Nx, MAX_CELLS), stride_for(Ny, MAX_CELLS), stride_for(Nz, MAX_CELLS)
    cap    = SLICE_CELLS if len(ks) > 1 else MAX_CELLS
    px, py = stride_for(Nx, cap), stride_for(Ny, cap)

    for name in PANELS:
        f = F[name]
        out = {"planes": [block_mean(p, py, px) for p in f["planes"]],
               "xw":     block_mean(f["xw"], sz, sy),
               "yw":     block_mean(f["yw"], sz, sx),
               "base":   f["base"]}
        if "mask_full" in f:
            m = f["mask_full"]
            out["mask"] = {"planes": [block_mean(a.astype(float), py, px) > 0.5
                                      for a in m["planes"]],
                           "xw": block_mean(m["xw"].astype(float), sz, sy) > 0.5,
                           "yw": block_mean(m["yw"].astype(float), sz, sx) > 0.5}
        F[name] = out

    #! ---- CELL-EDGE coordinates ----
    x0e, x1e = domain_edges(x, Lx, Nx)
    y0e, y1e = domain_edges(y, Ly, Ny)
    z0e, z1e = domain_edges(z, Lz, Nz)

    xe = np.linspace(x0e, x1e, n_blocks(Nx, sx) + 1)
    ye = np.linspace(y0e, y1e, n_blocks(Ny, sy) + 1)
    ze = np.linspace(z0e, z1e, n_blocks(Nz, sz) + 1)
    xp = np.linspace(x0e, x1e, n_blocks(Nx, px) + 1)
    yp = np.linspace(y0e, y1e, n_blocks(Ny, py) + 1)

    PX, PY    = np.meshgrid(xp, yp)
    WY, WZ_yz = np.meshgrid(ye, ze)
    WX, WZ_xz = np.meshgrid(xe, ze)

    plane_z = [float(z[int(k)]) for k in ks]

    pad_h = max(PAD_FRAC, max(TICK_PAD, AXIS_LABEL_PAD) + 0.05)

    def padded(lo, hi, pad=None):
        half = max(0.5 * (hi - lo), 1.0e-30) * (1.0 + (PAD_FRAC if pad is None else pad))
        mid  = 0.5 * (hi + lo)
        return (mid - half, mid + half)

    #! ---- physical box, then the same quantities in MPL order ----
    #! The limits, the box aspect and the tick edges all have to follow --up, so they are
    #! built once in physical space and pushed through to_mpl. Everything downstream
    #! (setup_axes3d, the ink fit) then needs no knowledge of the orientation at all.
    pbx = (float(xe[0]), float(xe[-1]))
    pby = (float(ye[0]), float(ye[-1]))
    pbz = (float(ze[0]), float(ze[-1]))
    mlx, mly, mlz = to_mpl(pbx, pby, pbz)                      #! box ranges, drawn order
    mwx, mwy, _   = to_mpl(float(xe[-1] if i_wall else xe[0]), #! near edges, drawn order
                           float(ye[-1] if j_wall else ye[0]),
                           float(ze[-1] if k_near else ze[0]))
    mnx, mny      = ORDER[0], ORDER[1]                         #! axis names, drawn order

    geom = {"PX": PX, "PY": PY, "WY": WY, "WZ_yz": WZ_yz, "WX": WX, "WZ_xz": WZ_xz,
            "plane_z": plane_z,
            #! physical, for the wireframe and the surf calls
            "box_x": pbx, "box_y": pby, "box_z": pbz,
            "x_wall": float(xe[-1] if i_wall else xe[0]),
            "y_wall": float(ye[-1] if j_wall else ye[0]),
            #! drawn order, for the axes, the ticks and the axis names
            "box_mx": mlx, "box_my": mly, "box_mz": mlz,
            "wall_mx": mwx, "wall_my": mwy,
            "name_mx": mnx, "name_my": mny,
            "xlim": padded(mlx[0], mlx[1], pad_h),
            "ylim": padded(mly[0], mly[1], pad_h),
            "zlim": padded(mlz[0], mlz[1]),
            "box": to_mpl(Lx, Ly, Lz)}

    #! ---- figure layout ----
    ncol = 2 if len(PANELS) > 1 else 1
    nrow = int(np.ceil(len(PANELS) / ncol))
    aspect  = ink_aspect(geom)
    draw_w  = PANEL_W - 2.0 * MARGIN_IN
    panel_h = (args.panel_h if args.panel_h is not None
               else float(np.clip(draw_w * aspect, 0.12 * draw_w, 2.00 * draw_w)))
    bind    = "fit" if args.panel_h is not None else "width"

    col_w = PANEL_W + CB_IN + COL_GAP
    fig_w = ncol * col_w
    fig_h = BOTTOM_IN + nrow * (panel_h + LABEL_IN) + TITLE_IN
    fig   = plt.figure(figsize=(fig_w, fig_h))

    def slot(ic, ir):
        x0_in = ic * col_w + MARGIN_IN
        y0_in = BOTTOM_IN + (nrow - 1 - ir) * (panel_h + LABEL_IN)
        return Bbox.from_bounds(x0_in / fig_w, y0_in / fig_h,
                                (PANEL_W - 2.0 * MARGIN_IN) / fig_w, panel_h / fig_h)

    base_slot = slot(0, 0)
    base_rect = fitted_rect((fig_w, fig_h), base_slot, geom, bind)

    for n, name in enumerate(PANELS):
        ic, ir = n % ncol, n // ncol
        sl = slot(ic, ir)
        rect = [base_rect[0] + (sl.x0 - base_slot.x0),
                base_rect[1] + (sl.y0 - base_slot.y0), base_rect[2], base_rect[3]]
        ax  = fig.add_axes(rect, projection="3d")
        cax = fig.add_axes([(ic * col_w + PANEL_W + 0.14) / fig_w, sl.y0 + 0.22 * sl.height,
                            0.24 / fig_w,                          0.56 * sl.height])
        render(fig, ax, name, F[name], geom, cax)
        fig.text(sl.x0 + 0.5 * sl.width, sl.y1 + 0.10 * LABEL_IN / fig_h, FIELDS[name]["label"],
                 fontsize=15, ha="center", va="bottom")

    fig.text(0.5, 1.0 - 0.45 * TITLE_IN / fig_h, time_label, fontsize=18, ha="center", va="center")

    outfile = f"{OUTDIR}/moments_3D_{step_idx:09d}.png"
    fig.savefig(outfile, dpi=DPI)
    plt.close(fig)
    plt.close("all")
    F, raw = None, None

    print(f"Saved {outfile}", flush=True)

comm.Barrier()
if rank == 0:
    print("\nDone.", flush=True)
    print("\n--------------------------\n\n", flush=True)