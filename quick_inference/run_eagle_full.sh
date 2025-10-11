#!/usr/bin/env bash
# ================================================================
# Gigapath Inference Runner
# ================================================================
# This script runs the full Gigapath feature extraction and inference
# pipeline on a directory of Whole Slide Images (WSIs).
#
# ---------------------------
# Usage:
#   ./run_gigapath.sh [optional arguments]
#
# Optional arguments passed after "$@" will be forwarded to the Python script.
#
# Example:
#   ./run_gigapath.sh --batch_size 32 --num_workers 8
#
# ---------------------------
# Before running:
#   1. Ensure the Python environment is active and paths are correct.
#   2. Verify that SLIDES_DIR points to a directory containing WSIs (.svs/.tif).
#   3. Ensure the Gigapath model checkpoints exist and are readable.
#   4. The output directory (OUTDIR) will be created automatically if missing.
# ================================================================


# --- Configurable defaults ---
PYTHON_BIN="~/anaconda3/envs/EAGLE/bin/python"     # Path to Python interpreter
SCRIPT_PATH="~/run_gigapath_full.py"               # Main Gigapath Python script

SLIDES_DIR="/media/tmp"                            # Input directory containing WSIs

# Paths to pre-trained model checkpoints
TILE_CKPT="~/EAGLE/checkpoints/gigapath_ft_checkpoint_tile_020.pth"
SLIDE_CKPT="~/EAGLE/checkpoints/gigapath_ft_checkpoint_slide_020.pth"

# Output configuration
OUTDIR="/media/test"                               # Directory where results will be saved
OUTNAME="test.csv"                                 # Output CSV filename

# --- Setup ---
# Create output directory if it does not exist
mkdir -p "${OUTDIR}"

# --- Run ---
# Execute the Gigapath Python script with all parameters.
# "$@" allows passing extra arguments from the command line to the Python script.
exec "${PYTHON_BIN}" "${SCRIPT_PATH}" \
  --slides_dir "${SLIDES_DIR}" \
  --outdir "${OUTDIR}" \
  --outname "${OUTNAME}" \
  --tile_checkpoint "${TILE_CKPT}" \
  --slide_checkpoint "${SLIDE_CKPT}" \
  "$@"

# ================================================================
# Notes:
# - Use absolute paths instead of ~ for HPC job submissions (expand with `realpath`).
# - Consider wrapping this in a SLURM submission script if using a GPU cluster.
# - To monitor progress: tail -f output.log (if redirected)
# ================================================================