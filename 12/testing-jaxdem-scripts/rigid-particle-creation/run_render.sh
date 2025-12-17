#!/bin/bash

# Usage check
if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <input_h5_file> <output_png_file> [base_pixels]"
    exit 1
fi

H5_INPUT="$1"
PNG_OUTPUT="$2"
BASE_PIXELS="${3:-1000}"

# Get the directory where the script is stored
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Create a safe temporary file with unique name (portable across macOS + Linux)
# GNU mktemp supports --suffix but macOS does not; use a template instead.
TEMP_VTP="$(mktemp "${TMPDIR:-/tmp}/jaxdem_render.XXXXXX.vtp")"

# Ensure cleanup happens on exit (success, error, or interrupt)
trap 'rm -f "$TEMP_VTP"' EXIT

# 1. Preprocess Data
# This runs in the standard python environment
echo "Running preprocessing on $H5_INPUT..."
METADATA=$(python "${SCRIPT_DIR}/preprocess.py" "$H5_INPUT" "$TEMP_VTP")

if [ $? -ne 0 ]; then
    echo "Preprocessing failed."
    exit 1
fi

echo "Metadata: $METADATA"

# 2. Render Scene
# This runs in pvbatch (ParaView environment)
echo "Running rendering -> $PNG_OUTPUT..."

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

# $METADATA expands to "DIM BOX_X BOX_Y [BOX_Z]"
#
# Headless Linux convenience:
# - If we're on Linux AND DISPLAY is unset, try xvfb-run if available.
# - On macOS (or any system with a working display), just run pvbatch directly.
# if [ "$(uname -s)" = "Linux" ] && [ -z "${DISPLAY:-}" ] && command -v xvfb-run >/dev/null 2>&1; then
if [ "$(uname -s)" = "Linux" ]; then
    echo "Headless Linux detected (DISPLAY unset). Using xvfb-run."
    xvfb-run -s "-screen 0 1600x1600x24" \
        "$PVBATCH" "${SCRIPT_DIR}/render.py" \
            "$TEMP_VTP" \
            "$PNG_OUTPUT" \
            $METADATA \
            "$BASE_PIXELS"
else
    "$PVBATCH" "${SCRIPT_DIR}/render.py" \
        "$TEMP_VTP" \
        "$PNG_OUTPUT" \
        $METADATA \
        "$BASE_PIXELS"
fi

if [ $? -ne 0 ]; then
    echo "Rendering failed."
    exit 1
fi

echo "Success."
