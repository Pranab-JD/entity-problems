#!/bin/bash
set -euo pipefail

### Make movies from fields_*.png and moments_*.png
### Usage:
###     bash make_movies.sh /path/to/plots/folder

### On LUMI
ml -q LUMI/25.09 FFmpeg/7.1.3-cpeGNU-25.09

folder="$1"

make_movie() {
    local pattern="$1"   # e.g. "fields_*.png"
    local output="$2"    # e.g. "${folder}/movie_fields.mp4"

    local tmpdir="${folder}/tmp_$(basename "${output%.mp4}")"

    rm -rf "$tmpdir"
    mkdir -p "$tmpdir"

    local i=0

    while IFS= read -r frame; do
        ln -s "$frame" "${tmpdir}/frame_$(printf "%06d" "$i").png"
        i=$((i + 1))
    done < <(
        find "$folder" -maxdepth 1 -type f -name "$pattern" |
        sort -V
    )

    if [ "$i" -eq 0 ]; then
        echo "Warning: no files found matching ${folder}/${pattern}, skipping." >&2
        rm -rf "$tmpdir"
        return
    fi

    ffmpeg -hide_banner -loglevel error -nostats -y \
        -framerate 3 \
        -i "${tmpdir}/frame_%06d.png" \
        -vf "scale=3840:-2,pad=ceil(iw/2)*2:ceil(ih/2)*2,setsar=1" \
        -c:v libx264 \
        -crf 18 \
        -threads 16 \
        -preset slow \
        -pix_fmt yuv420p \
        "$output"

    rm -rf "$tmpdir"

    echo "Created movie from $i frames: $output"
}

make_movie "fields_*.png"  "${folder}/Fields.mp4"
make_movie "moments_*.png" "${folder}/Moments.mp4"