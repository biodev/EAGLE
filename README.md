# Quick Inference

This module provides a minimal interface for running **Gigapath-based inference** using fine-tuned **EAGLE** model weights.  
It is designed for **rapid evaluation or testing** of pre-trained pathology foundation models without running the full IRT pipeline.

---

## 1. Overview

The quick inference workflow performs the following:

1. Reads a directory of whole-slide images (`.svs`, `.tif`, `.ndpi`, etc.)
2. Extracts tiles and generates embeddings using the fine-tuned tile-level encoder
3. Aggregates features using the slide-level MIL attention model
4. Produces slide-level predictions saved to a `.csv` file

Use this to validate exported weights, benchmark Gigapath encoders, or perform rapid inference on a few slides.

---

## 2. Directory Structure

```
quick_inference/
├── run_gigapath_full.py          # Main Python inference script
├── run_gigapath.sh               # Bash wrapper for easy execution
├── checkpoints/                  # Directory for model weights
├── example_slides/               # (Optional) Example WSIs for testing
└── outputs/                      # Default output folder for results
```

---

## 3. Environment Setup

Activate the same Conda environment used for EAGLE:

```bash
conda activate irt_env
```

Ensure dependencies are installed:

```bash
pip install -r ../requirements.txt
```

Optionally specify GPU(s):

```bash
export CUDA_VISIBLE_DEVICES=0
```

---

## 4. Downloading EAGLE Weights

The EAGLE fine-tuned Gigapath weights are hosted on Hugging Face:  
👉 https://huggingface.co/MCCPBR/EAGLE/tree/main

To download manually:

```bash
mkdir -p checkpoints
cd checkpoints

# Tile-level encoder
wget https://huggingface.co/MCCPBR/EAGLE/resolve/main/gigapath_ft_checkpoint_tile_020.pth

# Slide-level aggregator
wget https://huggingface.co/MCCPBR/EAGLE/resolve/main/gigapath_ft_checkpoint_slide_020.pth
```

Or use the Hugging Face CLI:

```bash
pip install -U "huggingface_hub>=0.21"
huggingface-cli download MCCPBR/EAGLE gigapath_ft_checkpoint_tile_020.pth --local-dir checkpoints
huggingface-cli download MCCPBR/EAGLE gigapath_ft_checkpoint_slide_020.pth --local-dir checkpoints
```

Expected layout:

```
quick_inference/checkpoints/gigapath_ft_checkpoint_tile_020.pth
quick_inference/checkpoints/gigapath_ft_checkpoint_slide_020.pth
```

---

## 5. Configure Paths

Edit `run_gigapath.sh` before running:

```bash
PYTHON_BIN="~/anaconda3/envs/EAGLE/bin/python"
SCRIPT_PATH="~/EAGLE/quick_inference/run_gigapath_full.py"

SLIDES_DIR="/path/to/slides"
TILE_CKPT="~/EAGLE/quick_inference/checkpoints/gigapath_ft_checkpoint_tile_020.pth"
SLIDE_CKPT="~/EAGLE/quick_inference/checkpoints/gigapath_ft_checkpoint_slide_020.pth"

OUTDIR="/path/to/output"
OUTNAME="results.csv"
```

---

## 6. Run Inference

Run locally:

```bash
cd quick_inference
bash run_gigapath.sh
```

Pass additional arguments supported by `run_gigapath_full.py`:

```bash
bash run_gigapath.sh --batch_size 32 --num_workers 8 --save_features
```

The script will:
- Create the output directory if it doesn’t exist  
- Process all slides in `SLIDES_DIR`  
- Save results to `${OUTDIR}/${OUTNAME}`  

---

## 7. Example Output

```csv
slide_id,predicted_score
slide_001.svs,0.9821
slide_002.svs,0.0332
slide_003.svs,0.7564
```

---

## 8. Optional: HPC (Slurm) Execution

For cluster environments, create `run_inference.slurm`:

```bash
#!/usr/bin/env bash
#SBATCH --job-name=eagle_infer
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

module load anaconda
source activate irt_env

bash run_gigapath.sh --batch_size 32 --num_workers 8
```

Submit with:

```bash
sbatch run_inference.slurm
```

---

## 9. Notes

- Always use **absolute paths** on HPC.  
- To log output:
  ```bash
  bash run_gigapath.sh > logs/inference_$(date +%Y%m%d_%H%M%S).log 2>&1
  ```
- Recommended GPU: ≥ 24 GB VRAM (A100 / H100).  
- Record the Hugging Face commit hash for reproducibility.

---

**References**  
Campanella et al., *Fine-Tuning Pathology Foundation Models for EGFR Prediction in Lung Adenocarcinoma*, 2024.  
Prov-GigaPath pretrained model: https://huggingface.co/prov-gigapath/prov-gigapath
