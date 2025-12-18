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


xvfb-run -s "-screen 0 1600x1600x24" \
  pvbatch "${SCRIPT_DIR}/render.py" \
    "$TEMP_VTP" \
    "$PNG_OUTPUT" \
    $METADATA \
    "$BASE_PIXELS"

if [ $? -ne 0 ]; then
    echo "Rendering failed."
    exit 1
fi

echo "Success."