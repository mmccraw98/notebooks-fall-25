#!/bin/bash

# Render an animated GIF from an H5 trajectory with time on the leading axis.
#
# Usage:
#   ./run_animation.sh <input_h5_file> <output_gif_file> <num_frames> [base_pixels] [fps]
#
# Notes:
# - If the H5 has more timesteps than num_frames, we evenly sample timesteps.
# - If fewer, we render all available timesteps.
# - Uses pvbatch for rendering (ParaView) and runs headless on Linux via xvfb-run.

set -euo pipefail

if [ "$#" -lt 3 ]; then
    echo "Usage: $0 <input_h5_file> <output_gif_file> <num_frames> [base_pixels] [fps]" >&2
    exit 1
fi

H5_INPUT="$1"
GIF_OUTPUT="$2"
NFRAMES="$3"
BASE_PIXELS="${4:-1000}"
FPS="${5:-15}"

# Get the directory where the script is stored
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Resolve pvbatch (allow override; helpful on macOS where it's inside ParaView.app)
PVBATCH="${PVBATCH:-pvbatch}"
if [ -x "$PVBATCH" ]; then
    :
elif command -v "$PVBATCH" >/dev/null 2>&1; then
    PVBATCH="$(command -v "$PVBATCH")"
elif [ -x "/Applications/ParaView.app/Contents/bin/pvbatch" ]; then
    PVBATCH="/Applications/ParaView.app/Contents/bin/pvbatch"
else
    echo "Error: pvbatch not found. Install ParaView or set PVBATCH=/path/to/pvbatch" >&2
    exit 127
fi

export PVBATCH

python "${SCRIPT_DIR}/animate.py" \
    "$H5_INPUT" \
    "$GIF_OUTPUT" \
    --num-frames "$NFRAMES" \
    --base-pixels "$BASE_PIXELS" \
    --fps "$FPS"


