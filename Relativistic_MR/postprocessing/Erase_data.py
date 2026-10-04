"""
Created on Mon Sep 28 2026

@author: Pranab JD, Claude AI

Shrink Entity ADIOS 'fields.*.bp' output by dropping unwanted variables

Usage
-----
    fields="/scratch/.../RMR/fields"

    srun python3 Erase_data.py "$fields"                   # DRY run/preview

    srun python3 -u Erase_data.py "$fields" --no-dry-run   # do it (in place)

"""

import os
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import glob
import shutil
import argparse
import numpy as np
from adios2 import Stream

try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank(); size = comm.Get_size(); HAVE_MPI = True
except Exception:
    comm = None; rank = 0; size = 1; HAVE_MPI = False

#! ============================================================
#! USER SETTINGS
#! ============================================================
FILE_GLOB = "fields.*.bp"       #! which files under the given directory to process

#! Choose ONE selection mode:
#!   DROP mode:  KEEP_VARS = None, list what to REMOVE in DROP_VARS
#!   KEEP mode:  set KEEP_VARS to the exact field vars to KEEP
DROP_VARS = ["fE1", "fE2", "fE3"]    #! e.g. remove B_z and in-plane currents
KEEP_VARS = None                     #! e.g. ["fB1", "fB2", "fJ3", "fN"] ; None -> use DROP_VARS

#! always preserved (coordinates + time + step) if present, regardless of the above
ALWAYS_KEEP = ["X1", "X2", "X3", "Time", "step"]

CHUNK_BYTES = 512 * 1024 * 1024  #! per-chunk memory budget for read/write (bytes)

TMP_SUFFIX = ".tmp_trim"         #! transient working copy; removed/renamed away, never left behind

#! ============================================================
#! Args  (dry-run is CLI; default is the SAFE dry run)
#! ============================================================
ap = argparse.ArgumentParser()
ap.add_argument("base", help="directory containing the .bp files")
ap.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=True,
                help="default: dry run (print only). Pass --no-dry-run to trim in place.")
args = ap.parse_args()
base    = args.base
dry_run = args.dry_run

#! ============================================================
#! Helpers
#! ============================================================
ITEMSIZE = {"float": 4, "float32": 4, "double": 8, "float64": 8,
            "int32_t": 4, "int64_t": 8, "uint32_t": 4, "uint64_t": 8,
            "int": 4, "long": 8, "char": 1, "signed char": 1, "unsigned char": 1}

def dir_size(path):
    if os.path.isfile(path):
        return os.path.getsize(path)
    tot = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                tot += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return tot

def parse_shape(shape_str):
    s = shape_str.strip().strip("{}[]")
    return [] if not s else [int(t) for t in s.replace(",", " ").split()]

def _fmt(nbytes):
    """Human-readable SI byte size (decimal, matches the GB=1e9 prints)."""
    x = float(nbytes)
    for unit in ("B", "KB", "MB", "GB", "TB", "PB"):
        if abs(x) < 1000.0 or unit == "PB":
            return f"{x:.2f} {unit}"
        x /= 1000.0

def _bulk_bytes(vinfo, names):
    """Sum of raw data bytes of the named variables (for dry-run projection)."""
    tot = 0
    for n in names:
        shp = parse_shape(vinfo[n].get("Shape", ""))
        it  = ITEMSIZE.get(vinfo[n].get("Type", "").strip(), 8)
        tot += it * (int(np.prod(shp)) if shp else 1)
    return tot

def resolve_keep(all_vars):
    always = {v for v in ALWAYS_KEEP if v in all_vars}
    if KEEP_VARS is not None:
        return {v for v in KEEP_VARS if v in all_vars} | always
    return (set(all_vars) - set(DROP_VARS)) | always     #! ALWAYS_KEEP overrides an accidental drop

def chunks_along_axis0(shape, itemsize):
    if not shape:                                        #! scalar
        yield None, None
        return
    n0 = shape[0]
    row_bytes = itemsize * int(np.prod(shape[1:])) if len(shape) > 1 else itemsize
    rows = max(1, int(CHUNK_BYTES // max(1, row_bytes)))
    for a in range(0, n0, rows):
        b = min(n0, a + rows)
        yield [a] + [0] * (len(shape) - 1), [b - a] + list(shape[1:])

#! ============================================================
#! Header (real runs only): inspect ONE file, print the var list + counts once.
#! Returns (all_vars, keep, drop) so the loop can flag any file that differs.
#! ============================================================
def inspect_vars(inpath):
    """Open one .bp, return (all_vars_sorted, keep_sorted, drop_sorted)."""
    with Stream(inpath, "r") as s:
        next(s.steps())
        vinfo = s.available_variables()
    all_vars = list(vinfo.keys())
    keep = resolve_keep(all_vars)
    drop = [v for v in all_vars if v not in keep]
    return sorted(all_vars), sorted(keep), sorted(drop)

def print_header(all_vars, keep, drop):
    """Single atomic print of the once-only header block."""
    msg = (
        "===========================================================\n"
        f"variables ({len(all_vars)}): {all_vars}\n"
        f"total: {len(all_vars)}   dropped ({len(drop)}): {drop}   "
        f"kept: {len(keep)}\n"
        "===========================================================\n"
    )
    print(msg, flush=True)

#! ============================================================
#! Process one BP file
#! ============================================================
def process_file(inpath, hdr_keep=None, hdr_drop=None):
    #! ---- inspect (no bulk read) ----
    with Stream(inpath, "r") as s:
        next(s.steps())
        vinfo = s.available_variables()
        try:
            ainfo = list(s.available_attributes().keys())
        except Exception:
            ainfo = []
    all_vars = list(vinfo.keys())
    keep = resolve_keep(all_vars)
    drop = [v for v in all_vars if v not in keep]
    sz_in = dir_size(inpath)

    #! ---- DRY RUN: print only (UNCHANGED detailed format) ----
    if dry_run:
        print(f"[{os.path.basename(inpath)}]  ({sz_in/1e9:.2f} GB)", flush=True)
        print(f"   current vars ({len(all_vars)}): {sorted(all_vars)}", flush=True)
        b_all  = _bulk_bytes(vinfo, all_vars) or 1
        b_drop = _bulk_bytes(vinfo, drop)
        freed  = sz_in * b_drop / b_all                 #! projected, scaled from file size
        print(f"   would keep ({len(keep)}): {sorted(keep)}", flush=True)
        print(f"   would drop ({len(drop)}): {sorted(drop)}", flush=True)
        print(f"   would clear ~{_fmt(freed)}\n", flush=True)
        return freed

    #! ---- REAL RUN: compact per-file block (atomic single print) ----
    #! flag (atomically) if THIS file's keep/drop differs from the header's,
    #! so a top-only summary never silently hides a differing file.
    if hdr_keep is not None and (sorted(keep) != hdr_keep or sorted(drop) != hdr_drop):
        print(f"[{os.path.basename(inpath)}]  WARNING: var set differs from "
              f"header -- keep={sorted(keep)} drop={sorted(drop)}\n", flush=True)

    if not drop:
        #! nothing to drop: report and leave file untouched (one atomic print)
        print(f"[{os.path.basename(inpath)}]\n"
              f"initial size: {sz_in/1e9:.2f} GB\n"
              f"nothing to drop; file unchanged\n", flush=True)
        return 0.0

    stem = inpath[:-3] if inpath.endswith(".bp") else inpath
    tmp  = stem + TMP_SUFFIX + ".bp"
    if os.path.exists(tmp):
        shutil.rmtree(tmp, ignore_errors=True)          #! clear a stale temp from a prior crash

    #! write trimmed copy to temp, chunked
    try:
        with Stream(inpath, "r") as sr:
            next(sr.steps())
            with Stream(tmp, "w") as sw:
                for aname in ainfo:                     #! attributes verbatim
                    try:
                        sw.write_attribute(aname, sr.read_attribute(aname))
                    except Exception:
                        pass
                for name in keep:                       #! kept variables, chunked
                    shape = parse_shape(vinfo[name].get("Shape", ""))
                    it    = ITEMSIZE.get(vinfo[name].get("Type", "").strip(), 8)
                    if not shape:
                        sw.write(name, np.asarray(sr.read(name)))
                        continue
                    for start, count in chunks_along_axis0(shape, it):
                        block = np.asarray(sr.read(name, start=start, count=count))
                        sw.write(name, block, shape=shape, start=start, count=count)
    except Exception as exc:
        print(f"[{os.path.basename(inpath)}]  WRITE FAILED: "
              f"{type(exc).__name__}: {exc} (original untouched)\n", flush=True)
        shutil.rmtree(tmp, ignore_errors=True)
        return 0.0

    #! verify temp BEFORE touching the original
    ok = True
    reason = ""
    try:
        with Stream(tmp, "r") as sc:
            next(sc.steps())
            have = set(sc.available_variables().keys())
        miss = [v for v in keep if v not in have]
        if miss:
            ok = False; reason = f"missing {miss}"
    except Exception as exc:
        ok = False; reason = f"cannot open temp: {exc}"
    sz_out = dir_size(tmp)
    if ok and sz_out >= sz_in:
        ok = False
        reason = f"not smaller ({sz_out/1e9:.2f} >= {sz_in/1e9:.2f} GB)"
    if not ok:
        print(f"[{os.path.basename(inpath)}]  VERIFY FAILED: {reason}; "
              f"original kept, temp removed\n", flush=True)
        shutil.rmtree(tmp, ignore_errors=True)
        return 0.0

    #! atomic-ish replace: same filename as before
    freed = 0.0
    try:
        shutil.rmtree(inpath)
        os.rename(tmp, inpath)
        freed = sz_in - sz_out
        #! ---- the requested compact per-file report (one atomic print) ----
        pct = 100 * (1 - sz_out / sz_in)
        print(f"[{os.path.basename(inpath)}]\n"
              f"initial size: {sz_in/1e9:.2f} GB\n"
              f"trimmed file size: {sz_out/1e9:.2f} GB (~{pct:.0f}% smaller)\n",
              flush=True)
    except Exception as exc:
        print(f"[{os.path.basename(inpath)}]  REPLACE FAILED: "
              f"{type(exc).__name__}: {exc} "
              f"(trimmed data is in {os.path.basename(tmp)}; original may be "
              f"gone -- check!)\n", flush=True)
    return freed

#! ============================================================
#! Main
#! ============================================================
files = sorted(glob.glob(os.path.join(base, FILE_GLOB)))
files = [f for f in files if not f.endswith(TMP_SUFFIX + ".bp")]   #! skip stale temps
if not files:
    raise SystemExit(f"No {FILE_GLOB} in {base}")

#! ---- header / mode banner (rank 0 only) ----
hdr_keep = hdr_drop = None
if rank == 0:
    mode = "DROP " + str(DROP_VARS) if KEEP_VARS is None else "KEEP " + str(KEEP_VARS)
    print(f"{len(files)} files; mode: {mode}; DRY_RUN={dry_run}\n", flush=True)
    if HAVE_MPI and size > len(files):
        print(f"NOTE: {size} tasks for {len(files)} files -> {size-len(files)} idle\n", flush=True)

    #! real runs: inspect ONE file up front and print the once-only var header.
    #! (dry-run keeps its original per-file detail, so no header there.)
    if not dry_run:
        try:
            all_v, hdr_keep, hdr_drop = inspect_vars(files[0])
            print_header(all_v, hdr_keep, hdr_drop)
        except Exception as exc:
            print(f"WARNING: could not read header vars from "
                  f"{os.path.basename(files[0])}: {exc}\n", flush=True)

#! broadcast the header keep/drop sets so every rank can flag a differing file
if HAVE_MPI:
    hdr_keep = comm.bcast(hdr_keep, root=0)
    hdr_drop = comm.bcast(hdr_drop, root=0)
    comm.Barrier()                                  #! header prints before file blocks

local_freed = 0.0
for f in files[rank::size]:
    try:
        local_freed += process_file(f, hdr_keep, hdr_drop) or 0.0
    except Exception as exc:
        print(f"SKIP {os.path.basename(f)}: {type(exc).__name__}: {exc}", flush=True)

if HAVE_MPI:
    total_freed = comm.reduce(local_freed, op=MPI.SUM, root=0)
    comm.Barrier()
else:
    total_freed = local_freed

if rank == 0:
    verb = "would clear (projected)" if dry_run else "cleared"
    print(f"\nTotal storage {verb}: {_fmt(total_freed)}", flush=True)
    print("Done.", flush=True)