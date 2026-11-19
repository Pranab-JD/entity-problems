"""
Created on Thu Nov 19 2026

@author: Pranab JD, Claude AI

Decaying kinetic turbulence -- non-ideal electric field, energy transfer, and the RMS current.

Usage:

    input="/scratch/project_465002528/pjd/turbulence/fields/"
    output="/scratch/project_465002528/pjd/turbulence/plots/"

    srun python3 ../postprocessing/Plot_EB_JE.py "$input" "$output" --Lx 256.0 --Ly 256.0

    Required:
        --Lx        box length along x in code units (d_e)
        --Ly        box length along y in code units (d_e)
        --Lz        box length along z in code units (d_e); REQUIRED for a 3D run,
                    rejected for a 2D run

    Optional:
        --t-eddy    outer-scale eddy turnover time in code units; copy the value
                    "Eddy turnover time [1/w_p1]" printed by the pgen
        --pct       percentile used for the colour limits (default: 99.5)
        --per-frame rescale colours independently for every snapshot
                    (default: fixed scale shared by all frames)

Panels
------
    E.B / (|E||B|)  cosine of the angle between E and B. Zero where the field is ideal
                    (E = -u x B is perpendicular to B); nonzero where it is not, i.e. in
                    current sheets and reconnection sites. Where |E| is tiny the angle is
                    noise, so pixels with |E| below E_FLOOR x (the plane's RMS |E|) are blanked.
    J rms vs time   RMS over the box of J_Z (out of plane, along the guide field) and of J_XY
                    (in plane). The time of maximum J_Z rms (t_peak, fully developed turbulence) 
                    is marked, and a grey line marks the time of the snapshot. 
    J_par E_par     the part of J.E along the LOCAL B: parallel acceleration and heating.
    J_perp.E_perp   the part across the local B: work on the bulk flow and perpendicular heating.
                    J.E = J_par E_par + J_perp.E_perp exactly; the two share one colour scale.

2D/3D compatibility
-------------------
    Entity writes field arrays with the SLOWEST axis first:
        2D run : (Ny, Nx)
        3D run : (Nz, Ny, Nx)        [axis0 = z, axis1 = y, axis2 = x]
    The dimensionality is taken from the presence of the X3 coordinate array, not from
    the shape of a field, and a 3D dump is cut at the mid-plane in z (axis 0).
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
N_SAMPLE = 8          #! snapshots sampled when building the shared colour scale
E_FLOOR  = 0.05       #! blank E.B/(|E||B|) where |E| < E_FLOOR x RMS |E| of the plane (angle is noise there)
RMS_CHUNK_Z = 16      #! 3D: z-planes read at a time for the box RMS of J (memory ~ 3 x RMS_CHUNK_Z x Nx x Ny values)

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
parser.add_argument("base",        type=str,   help="Directory with fields.NNNNNNNNN.bp files")
parser.add_argument("outdir",      type=str,   help="Output directory for PNG plots")
parser.add_argument("--Lx",        type=float, required=True, help="Box length along x in code units")
parser.add_argument("--Ly",        type=float, required=True, help="Box length along y in code units")
parser.add_argument("--Lz",        type=float, default=None,  help="Box length along z in code units (required for 3D runs)")
parser.add_argument("--t-eddy",    type=float, default=None,  dest="t_eddy", help="Outer eddy turnover time [1/w_p1] from the pgen printout")
parser.add_argument("--pct",       type=float, default=99.5,  help="Percentile for colour limits")
parser.add_argument("--per-frame", action="store_true", dest="per_frame", help="Rescale colours per snapshot instead of using one fixed scale")

args      = parser.parse_args()
base      = args.base
outdir    = args.outdir
Lx        = args.Lx
Ly        = args.Ly
Lz        = args.Lz
t_eddy    = args.t_eddy
pct       = args.pct
per_frame = args.per_frame

if Lx <= 0.0:
    parser.error("--Lx must be positive")
if Ly <= 0.0:
    parser.error("--Ly must be positive")
if Lz is not None and Lz <= 0.0:
    parser.error("--Lz must be positive")
if not (50.0 < pct <= 100.0):
    parser.error("--pct must be in (50, 100]")
if t_eddy is not None and t_eddy <= 0.0:
    parser.error("--t-eddy must be positive")

if rank == 0:
    os.makedirs(outdir, exist_ok=True)
comm.Barrier()

#! ============================================================
#! Panel definition
#!   label -> (kind, cmap, group)
#!   kind  = "div" : diverging, symmetric limits about 0 (signed quantities)
#!   group : panels sharing a group share one colour scale
#! ============================================================
EB_LABEL    = r"$\mathbf{E}\cdot\mathbf{B}\,/\,(|\mathbf{E}||\mathbf{B}|)$"
JEPAR_LABEL = r"$J_\parallel E_\parallel$"
JEPER_LABEL = r"$\mathbf{J}_\perp\cdot\mathbf{E}_\perp$"

PANELS = [
    (EB_LABEL,    "div", "seismic", "EB_angle"),
    (JEPAR_LABEL, "div", "seismic", "JdotE"),
    (JEPER_LABEL, "div", "seismic", "JdotE"),
]
MAP_POSITIONS    = [(0, 0), (1, 0), (1, 1)]     #! (row, column) of each map in PANELS
SERIES_POSITION  = (0, 1)                       #! J rms time series, top right

GROUP_KIND = {}
for _label, _kind, _cmap, _group in PANELS:
    GROUP_KIND[_group] = _kind

#! ============================================================
#! Helpers
#! ============================================================
files = sorted(glob.glob(f"{base}/fields.*.bp"))

if len(files) == 0:
    if rank == 0:
        print(f"ERROR: no fields.*.bp in {base}", flush=True)
    raise SystemExit(1)

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

def grid_sizes(stream):
    """
    (Nx, Ny, Nz) from the coordinate arrays; Nz is None for a 2D dump. The coordinate
    arrays are 1D and tiny, so reading them whole costs nothing and settles the
    dimensionality without touching a field variable.
    """
    Nx = np.asarray(stream.read("X1")).size
    Ny = np.asarray(stream.read("X2")).size
    try:
        Nz = np.asarray(stream.read("X3")).size
    except Exception:
        Nz = None
    return Nx, Ny, (Nz if (Nz is not None and Nz > 1) else None)

def probe_dimensionality(fname):
    with Stream(fname, "r") as s:
        next(s.steps())
        _, _, Nz = grid_sizes(s)
    return (Nz is not None), (int(Nz) if Nz is not None else 0)

is_3d, Nz_global = (None, None)
if rank == 0:
    is_3d, Nz_global = probe_dimensionality(files[0])
is_3d, Nz_global = comm.bcast((is_3d, Nz_global), root=0)

if is_3d and Lz is None:
    if rank == 0:
        print("ERROR: this is a 3D run (X3 present); --Lz is required", flush=True)
    raise SystemExit(1)
if (not is_3d) and Lz is not None:
    if rank == 0:
        print("ERROR: this is a 2D run; --Lz is not applicable", flush=True)
    raise SystemExit(1)

if rank == 0:
    print(f"Found {len(files)} files", flush=True)
    print(f"    Dimensionality      = {'3D' if is_3d else '2D'}", flush=True)
    print(f"    Box length L_x [d_e] = {Lx}", flush=True)
    print(f"    Box length L_y [d_e] = {Ly}", flush=True)
    if is_3d:
        print(f"    Box length L_z [d_e] = {Lz}", flush=True)
        print(f"    Cut plane           = z = {0.5 * Lz:g} (k = {Nz_global // 2} of {Nz_global})", flush=True)
    if t_eddy is not None:
        print(f"    Eddy turnover  [1/w_p] = {t_eddy}", flush=True)
    print(f"    Colour scale        = {'per frame' if per_frame else 'fixed across all frames'}", flush=True)
    print(f"    E.B angle blanked where |E| < {E_FLOOR:g} x RMS |E|", flush=True)
    print(" ", flush=True)

files_local = files[rank::size]

def read_plane(stream, name, Nx, Ny, Nz, k, missing):
    """
    One x-y plane as (Ny, Nx): rows = y, columns = x, which is what imshow expects
    with origin='lower' and extent=[0, Lx, 0, Ly].
      2D: the whole (Ny, Nx) array.
      3D: ONLY the k-th plane, as a hyperslab -- start/count, not a numpy slice.

    Returns None, and records the name in 'missing', if the variable is absent or does
    not have the expected plane size. A missing variable is not an error.
    """
    try:
        if Nz is None:
            a = np.asarray(stream.read(name))
        else:
            a = np.asarray(stream.read(name, start=[int(k), 0, 0], count=[1, int(Ny), int(Nx)]))
    except Exception:
        missing.add(name)
        return None

    if a is None or a.size != int(Nx) * int(Ny):
        missing.add(name)
        return None

    return a.reshape(int(Ny), int(Nx))

def load_panels(fname):
    """
    Everything one figure needs, from one open of the file. Used BOTH by the colour-
    scale scan and by the plotting loop, so the two can never disagree.

    Returns (panels, t_code, x, y, missing); a panel value of None means "not in the
    dump, leave it blank".
    """
    missing = set()

    with Stream(fname, "r") as s:
        next(s.steps())

        x      = np.asarray(s.read("X1"))
        y      = np.asarray(s.read("X2"))
        t_code = read_time(s)

        Nx, Ny, Nz = grid_sizes(s)
        k_mid = (Nz // 2) if Nz is not None else 0      #! mid-plane in z, 3D only

        Ex = read_plane(s, "fE1", Nx, Ny, Nz, k_mid, missing)
        Ey = read_plane(s, "fE2", Nx, Ny, Nz, k_mid, missing)
        Ez = read_plane(s, "fE3", Nx, Ny, Nz, k_mid, missing)

        Bx = read_plane(s, "fB1", Nx, Ny, Nz, k_mid, missing)
        By = read_plane(s, "fB2", Nx, Ny, Nz, k_mid, missing)
        Bz = read_plane(s, "fB3", Nx, Ny, Nz, k_mid, missing)

        Jx = read_plane(s, "fJ1", Nx, Ny, Nz, k_mid, missing)
        Jy = read_plane(s, "fJ2", Nx, Ny, Nz, k_mid, missing)
        Jz = read_plane(s, "fJ3", Nx, Ny, Nz, k_mid, missing)

    have_E = (Ex is not None) and (Ey is not None) and (Ez is not None)
    have_B = (Bx is not None) and (By is not None) and (Bz is not None)
    have_J = (Jx is not None) and (Jy is not None) and (Jz is not None)

    #! cos(angle between E and B) = E.B / (|E||B|), blanked where |E| is too small to define an angle
    EB_angle = None
    if have_E and have_B:
        E_mag = np.sqrt(Ex**2 + Ey**2 + Ez**2)
        B_mag = np.sqrt(Bx**2 + By**2 + Bz**2)
        E_rms = float(np.sqrt(np.mean(E_mag**2)))
        valid = (E_mag > E_FLOOR * E_rms) & (B_mag > 0.0)
        EB_angle = np.full(E_mag.shape, np.nan)
        EB_angle[valid] = (Ex * Bx + Ey * By + Ez * Bz)[valid] / (E_mag[valid] * B_mag[valid])

    #! J.E and its split along / across the local magnetic field direction b = B/|B|
    JdotE, JE_par, JE_perp = None, None, None
    if have_J and have_E:
        JdotE = Jx * Ex + Jy * Ey + Jz * Ez
        if have_B:
            B_mag = np.sqrt(Bx**2 + By**2 + Bz**2)
            with np.errstate(divide="ignore", invalid="ignore"):
                bx, by, bz = Bx / B_mag, By / B_mag, Bz / B_mag
            J_par   = Jx * bx + Jy * by + Jz * bz
            E_par   = Ex * bx + Ey * by + Ez * bz
            JE_par  = J_par * E_par                         #! J_par E_par
            JE_perp = JdotE - JE_par                        #! J_perp . E_perp, so the two add up to J.E exactly

    panels = {
        EB_LABEL:    EB_angle,
        JEPAR_LABEL: JE_par,
        JEPER_LABEL: JE_perp,
    }
    return panels, t_code, x, y, missing

def panel_extremum(data, kind):
    if data is None:
        return None
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return None
    v = np.percentile(np.abs(finite), pct) if kind == "div" else np.percentile(finite, pct)
    return float(v)

def limits_from_extrema(acc):
    limits = {}
    for group, kind in GROUP_KIND.items():
        v = acc[group]
        if v is None or not np.isfinite(v) or v <= 0.0:
            v = 1.0
        limits[group] = (-v, v) if kind == "div" else (0.0, v)
    return limits

def accumulate(acc, panels):
    for label, kind, _, group in PANELS:
        v = panel_extremum(panels[label], kind)
        if v is None:
            continue
        acc[group] = v if acc[group] is None else max(acc[group], v)
    return acc

def scan_limits(all_files):
    """
    One colour scale for the whole movie, built on rank 0 from N_SAMPLE snapshots
    spread evenly over the run, then broadcast.

    Sampling rather than using the first file matters: E can start at zero in this pgen
    (no velocity perturbation), so t = 0 limits would saturate every later frame.
    """
    payload = None
    if rank == 0:
        idx = np.unique(np.linspace(0, len(all_files) - 1, min(N_SAMPLE, len(all_files))).astype(int))
        acc = {group: None for group in GROUP_KIND}
        missing = set()
        for i in idx:
            panels, _, _, _, miss = load_panels(all_files[i])
            missing |= miss
            accumulate(acc, panels)

        limits = limits_from_extrema(acc)

        if missing:
            print(f"  [warn] not in the dump, panels left blank: {', '.join(sorted(missing))}", flush=True)
        print(f"Fixed colour limits from {len(idx)} sampled snapshots (p{pct}):", flush=True)
        for group in GROUP_KIND:
            tag = "" if acc[group] is not None else "   (no data)"
            print(f"    {group:12s} [{limits[group][0]:+.4g}, {limits[group][1]:+.4g}]{tag}", flush=True)
        print(" ", flush=True)

        payload = limits

    return comm.bcast(payload, root=0)

def frame_limits(panels):
    acc = {group: None for group in GROUP_KIND}
    accumulate(acc, panels)
    return limits_from_extrema(acc)

def set_equidistant_ticks(ax, xlo, xhi, ylo, yhi, n=N_TICKS):
    xt = np.linspace(xlo, xhi, n)
    yt = np.linspace(ylo, yhi, n)
    ax.set_xticks(xt)
    ax.set_yticks(yt)
    ax.set_xticklabels([f"{v:g}" for v in xt])
    ax.set_yticklabels([f"{v:g}" for v in yt])
    ax.minorticks_off()                     #! "no more, no less" -> no minor ticks either

def current_rms(fname):
    """
    (t_code, rms J_Z, rms J_XY) over the WHOLE box: J_Z^rms = sqrt(<J_z^2>), J_XY^rms = sqrt(<J_x^2 + J_y^2>).
    2D reads the full arrays; 3D reads RMS_CHUNK_Z planes at a time. NaN if J is not in the dump.
    """
    with Stream(fname, "r") as s:
        next(s.steps())
        t_code = read_time(s)
        Nx, Ny, Nz = grid_sizes(s)
        sum_z_sq, sum_xy_sq, n_points = 0.0, 0.0, 0
        try:
            if Nz is None:
                slabs = [(None, None)]
            else:
                slabs = [([k0, 0, 0], [min(RMS_CHUNK_Z, Nz - k0), int(Ny), int(Nx)]) for k0 in range(0, Nz, RMS_CHUNK_Z)]
            for start, count in slabs:
                if start is None:
                    Jx, Jy, Jz = (np.asarray(s.read(name), dtype=np.float64) for name in ("fJ1", "fJ2", "fJ3"))
                else:
                    Jx, Jy, Jz = (np.asarray(s.read(name, start=start, count=count), dtype=np.float64) for name in ("fJ1", "fJ2", "fJ3"))
                sum_z_sq  += float(np.sum(Jz**2))
                sum_xy_sq += float(np.sum(Jx**2 + Jy**2))
                n_points  += Jz.size
        except Exception:
            return t_code, float("nan"), float("nan")
    if n_points == 0:
        return t_code, float("nan"), float("nan")
    return t_code, np.sqrt(sum_z_sq / n_points), np.sqrt(sum_xy_sq / n_points)

#* every rank does its own files, then all ranks get the whole series, sorted by time
rms_local  = [current_rms(fname) for fname in files_local]
rms_all    = sorted([row for rows in comm.allgather(rms_local) for row in rows], key=lambda row: row[0])
rms_time   = np.array([row[0] for row in rms_all])
rms_J_z    = np.array([row[1] for row in rms_all])
rms_J_xy   = np.array([row[2] for row in rms_all])
has_rms    = bool(np.any(np.isfinite(rms_J_z)))
t_peak     = float(rms_time[np.nanargmax(rms_J_z)]) if has_rms else float("nan")   #! time of maximum J_Z rms

if rank == 0:
    if has_rms:
        print(f"J_Z rms peaks at t = {t_peak:.4g}  (t c/L_x = {t_peak / Lx:.4g}): rms J_Z = {np.nanmax(rms_J_z):.4g}", flush=True)
        if t_peak == rms_time[-1]:
            print("  [note] the peak is at the LAST snapshot: J_Z rms may still be growing", flush=True)
    else:
        print("  [warn] fJ1/fJ2/fJ3 not in the dumps: J rms panel left blank", flush=True)
    print(" ", flush=True)

#! ============================================================
#! Shared colour scale (unless --per-frame)
#! ============================================================
fixed_limits = None if per_frame else scan_limits(files)
comm.Barrier()

#! ============================================================
#! Loop over assigned files
#! ============================================================
x_lo, x_hi = 0.0, Lx
y_lo, y_hi = 0.0, Ly

for fname in files_local:

    step_idx = step_from_fname(fname)

    panels, t_code, x, y, missing = load_panels(fname)

    #! sanity check: the file should agree with --Lx/--Ly to within about one cell
    if rank == 0 and fname == files_local[0]:
        dx_est = (x.max() - x.min()) / max(x.size - 1, 1)
        dy_est = (y.max() - y.min()) / max(y.size - 1, 1)
        if abs(x.max() - x.min() + dx_est - Lx) > 2.0 * dx_est or \
           abs(y.max() - y.min() + dy_est - Ly) > 2.0 * dy_est:
            print(f"  [warn] file extent is x:[{x.min():g},{x.max():g}] y:[{y.min():g},{y.max():g}] "
                  f"but --Lx={Lx:g} --Ly={Ly:g}", flush=True)

    limits = fixed_limits if fixed_limits is not None else frame_limits(panels)

    #! ========================================================
    #! Time labels. t_code is t w_p; t c/L_x is the light crossing count.
    #! ========================================================
    if not np.isnan(t_code):
        time_label = f"$t\\,\\omega_{{p}} = {t_code:.1f}$   |   $t\\,c/L_x = {t_code / Lx:.3f}$"
        if t_eddy is not None:
            time_label += f"   |   $t/t_{{\\rm eddy}} = {t_code / t_eddy:.2f}$"
        if is_3d:
            time_label += f"   |   $z = {0.5 * Lz:g}\\,d_e$"
    else:
        time_label = f"step {step_idx:09d}   (time not found -- check TIME_KEY)"

    #! ========================================================
    #! Plot
    #! ========================================================
    fig, axs = plt.subplots(2, 2, figsize=(11, 9))

    for (row, column), (label, kind, cmap, group) in zip(MAP_POSITIONS, PANELS):

        ax = axs[row, column]

        data = panels[label]

        ax.set_title(label)
        ax.tick_params(axis="both", which="major", labelsize=10, length=6)
        ax.set_xlabel(r"$x\ [d_e]$")
        ax.set_ylabel(r"$y\ [d_e]$")
        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(y_lo, y_hi)
        ax.set_aspect("equal")

        #! exactly 5 equidistant ticks + labels on each axis of this panel
        set_equidistant_ticks(ax, x_lo, x_hi, y_lo, y_hi)

        if data is None:
            ax.text(0.5, 0.5, "not in dump", transform=ax.transAxes,
                    ha="center", va="center", fontsize=11, color="0.5")
            continue

        vmin, vmax = limits[group]

        cmap_obj = plt.get_cmap(cmap).copy()
        cmap_obj.set_bad("0.85")                    #! blanked pixels (|E| too small) in light grey
        im = ax.imshow(np.ma.masked_invalid(data), origin="lower", aspect="equal",
                       extent=[x_lo, x_hi, y_lo, y_hi], cmap=cmap_obj, vmin=vmin, vmax=vmax)

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    #* J rms time series (Franci et al. 2022, Fig. 1a): whole run, t_peak marked, this snapshot in grey
    ax = axs[SERIES_POSITION]
    ax.set_title(r"RMS current density")
    if has_rms:
        ax.plot(rms_time / Lx, rms_J_xy, color="blue", lw=1.6, label=r"$J_{\rm XY}^{\rm rms}$ (in plane)")
        ax.plot(rms_time / Lx, rms_J_z,  color="red",  lw=1.6, label=r"$J_{\rm Z}^{\rm rms}$ (out of plane)")
        ax.axvline(t_peak / Lx, color="green", lw=1.2, ls="--")
        ax.plot([t_peak / Lx], [np.nanmax(rms_J_z)], "o", color="green", ms=5)
        ax.annotate(r"$t_{\rm peak}$", (t_peak / Lx, np.nanmax(rms_J_z)), textcoords="offset points", xytext=(5, -16), color="green", fontsize=11)
        if not np.isnan(t_code):
            ax.axvline(t_code / Lx, color="0.5", lw=1.0)
        ax.set_ylim(bottom=0.0)
        ax.legend(fontsize=10, loc="lower right")
    else:
        ax.text(0.5, 0.5, "J not in dump", transform=ax.transAxes, ha="center", va="center", fontsize=11, color="0.5")
    ax.set_xlabel(r"$t\,c/L_x$")
    ax.set_ylabel(r"$J^{\rm rms}$ [code units]")
    ax.tick_params(axis="both", which="major", labelsize=10, length=6)

    fig.suptitle(time_label, fontsize=16, y=1.00)
    fig.tight_layout()

    outfile = f"{outdir}/EB_JE_2D_{step_idx:09d}.png"
    fig.savefig(outfile, dpi=150)
    plt.close(fig)

    print(f"Saved {outfile}", flush=True)

comm.Barrier()

if rank == 0:
    print("\nDone.", flush=True)
    print("\n--------------------------\n\n", flush=True)