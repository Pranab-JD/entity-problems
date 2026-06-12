#!/bin/bash
# =============================================================================
# make_movies.sh — combine field and moment PNG frames into MP4 movies
#
# Usage:
#   bash make_movies.sh <plots_dir> [fps]
#
# Arguments:
#   plots_dir   directory containing fields_*.png and moments_*.png
#   fps         frames per second (default: 5)
#
# Output (saved in plots_dir):
#   fields_movie.mp4   — from fields_NNNNNNNNN.png
#   moments_movie.mp4  — from moments_NNNNNNNNN.png
#
# =============================================================================

set -euo pipefail

### On LUMI
ml -q LUMI/25.09 FFmpeg/7.1.3-cpeGNU-25.09

#! ============================================================
#! Arguments
#! ============================================================
if [[ $# -lt 1 ]]; then
    echo "Usage: bash make_movies.sh <plots_dir> [fps]"
    exit 1
fi

PLOTS_DIR="${1%/}"
FPS="${2:-5}"

#! ============================================================
#! Check ffmpeg is available
#! ============================================================
if ! command -v ffmpeg &> /dev/null; then
    echo "ERROR: ffmpeg not found."
    echo "       On LUMI: ml -q LUMI/25.09 FFmpeg/7.1.3-cpeGNU-25.09"
    exit 1
fi

#! ============================================================
#! Helper: build one movie from a glob pattern.
#!
#!   Uses a symlinked tmp directory (same approach as B_E_movie.sh)
#!   so ffmpeg gets a clean sequential frame_%06d.png sequence
#!   regardless of the original numbering gaps.
#!
#!   $1  glob pattern  e.g. "fields_*.png"
#!   $2  output MP4 filename
#! ============================================================
make_movie() {
    local pattern="$1"
    local outfile="$2"
    local prefix
    prefix=$(basename "$outfile" .mp4)

    # Count matching files
    local count
    count=$(find "$PLOTS_DIR" -maxdepth 1 -type f -name "$pattern" | wc -l)

    if [[ $count -eq 0 ]]; then
        echo "  No files matching '$pattern' in $PLOTS_DIR — skipping."
        return
    fi

    echo "  Found $count frames matching '$pattern'"

    # Build a clean sequential symlink directory (B_E_movie.sh approach).
    # sort -V gives version-sort (numeric within strings) so that
    # fields_000000009.png < fields_000000251.png correctly.
    local tmpdir="${PLOTS_DIR}/tmp_${prefix}"
    rm -rf "$tmpdir"
    mkdir -p "$tmpdir"

    local i=0
    while IFS= read -r frame; do
        ln -s "$frame" "${tmpdir}/frame_$(printf "%06d" "$i").png"
        i=$((i + 1))
    done < <(
        find "$PLOTS_DIR" -maxdepth 1 -type f -name "$pattern" | sort -V
    )

    echo "  Writing $outfile ..."

    ffmpeg -hide_banner -loglevel error -nostats -y \
        -framerate "$FPS" \
        -i "${tmpdir}/frame_%06d.png" \
        -vf "scale=3840:-2,pad=ceil(iw/2)*2:ceil(ih/2)*2,setsar=1" \
        -c:v libx264 \
        -crf 18 \
        -threads 16 \
        -preset slow \
        -pix_fmt yuv420p \
        "$outfile"

    rm -rf "$tmpdir"
    echo "  Done: $outfile  ($i frames)"
    echo ""
}

#! ============================================================
#! Build movies
#! ============================================================
echo "============================================"
echo "  plots dir : $PLOTS_DIR"
echo "  fps       : $FPS"
echo "============================================"
echo ""

make_movie "fields_*.png"  "$PLOTS_DIR/fields_movie.mp4"
make_movie "moments_*.png" "$PLOTS_DIR/moments_movie.mp4"

echo "============================================"
echo "  All done."
echo "============================================"