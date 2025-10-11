#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standalone inference pipeline for GigaPath slides.

Key differences vs. run_gigapath_file_path_memory_optimized:
  * Accepts a directory of .svs files instead of a manifest CSV.
  * Inlines all project-specific helpers (no external repo imports).
  * Emits the same CSV schema as the original pipeline.
"""

import os

# Conservative allocator + disable cuCIM caches
os.environ.setdefault("CUCIM_CACHE_TYPE", "none")
os.environ.setdefault("CUCIM_CACHE_DEVICE_MEMORY_SIZE", "0")
os.environ.setdefault("CUCIM_CACHE_HOST_MEMORY_SIZE", "0")
os.environ.setdefault("CUCIM_CACHE_NVCOMP_DEVICE_MEMORY_SIZE", "0")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:64")

import sys
import gc
import cv2
import random
import argparse
import warnings
from pathlib import Path
from datetime import datetime
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
import timm
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms as transforms

from PIL import Image, ImageDraw
from skimage.filters import threshold_otsu
from skimage.morphology import binary_erosion, binary_dilation, label

import openslide
from tqdm.auto import tqdm

# ======================= constants =======================
SVS_EXTENSIONS = (".svs", ".tif", ".tiff")
DEFAULT_TILE_MODEL = "hf_hub:prov-gigapath/prov-gigapath"
ISYNTAX_MPP = 0.25
MAX_PIXEL_DIFFERENCE = 0.2
THUMB_SUBDIR = "thumbnails"

# ======================= utility helpers =======================
def fix_state_dict(state_dict):
    return OrderedDict(
        (k[7:] if k.startswith("module.") else k, v) for k, v in state_dict.items()
    )


def free_vram_gb(idx=None) -> float:
    try:
        free, _ = torch.cuda.mem_get_info(idx)
        return free / (1024**3)
    except Exception:
        return 0.0


def pick_device(min_free_gb: float, strict: bool = False) -> torch.device:
    if not torch.cuda.is_available():
        print("[device] CUDA not available -> CPU")
        return torch.device("cpu")

    n = torch.cuda.device_count()
    best, best_free = None, -1.0
    per_gpu = {}
    for i in range(n):
        f = free_vram_gb(i)
        per_gpu[i] = f
        if f > best_free:
            best, best_free = i, f

    print(
        "[device] gpu_min_free_gb={:.2f} | free_gb={} | best=cuda:{} ({:.2f} GiB)".format(
            float(min_free_gb),
            {k: round(v, 2) for k, v in per_gpu.items()},
            best,
            best_free,
        )
    )

    if best is None:
        print("[device] No visible CUDA devices -> CPU")
        return torch.device("cpu")

    if best_free >= float(min_free_gb):
        torch.cuda.set_device(best)
        return torch.device(f"cuda:{best}")

    if strict:
        print("[device] Below threshold and strict=True -> CPU")
        return torch.device("cpu")

    print("[device] Below threshold but strict=False -> using best GPU anyway")
    torch.cuda.set_device(best)
    return torch.device(f"cuda:{best}")


# ======================= model definitions =======================
class AttnNetGated(nn.Module):
    def __init__(self, L=1024, D=256, dropout=False, n_tasks=1):
        super().__init__()
        self.attention_a = [nn.Linear(L, D), nn.Tanh()]
        self.attention_b = [nn.Linear(L, D), nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))
        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        self.attention_c = nn.Linear(D, n_tasks)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = self.attention_c(a.mul(b))
        return A, x


class GMA(nn.Module):
    def __init__(
        self,
        ndim=1024,
        gate=True,
        size_arg="big",
        dropout=False,
        n_classes=2,
    ):
        super().__init__()
        size_dict = {"small": [ndim, 512, 256], "big": [ndim, 512, 384]}
        size = size_dict[size_arg]

        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        if dropout:
            fc.append(nn.Dropout(0.25))
        fc.extend([nn.Linear(size[1], size[1]), nn.ReLU()])
        if dropout:
            fc.append(nn.Dropout(0.25))
        attention_net = AttnNetGated(L=size[1], D=size[2], dropout=dropout, n_tasks=1)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        self.classifier = nn.Linear(size[1], n_classes)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                m.bias.data.zero_()

    def forward(self, h, attention_only=False):
        A, h = self.attention_net(h)
        A = torch.transpose(A, 1, 0)
        if attention_only:
            return A[0]
        A_raw = A.detach().cpu().numpy()[0]
        w = self.classifier.weight.detach()
        sign = torch.mm(h.detach(), w.t()).cpu().numpy()
        A = F.softmax(A, dim=1)
        M = torch.mm(A, h)
        logits = self.classifier(M)
        return A_raw, sign, logits


def load_tile_model():
    try:
        return timm.create_model(DEFAULT_TILE_MODEL, pretrained=True)
    except Exception as exc:
        msg = str(exc).lower()
        if "unauthorized" in msg or "gated" in msg or "401" in msg:
            raise RuntimeError(
                "Failed to download the GigaPath backbone from Hugging Face. "
                "Pass --hf_token or set HF_TOKEN / HUGGING_FACE_HUB_TOKEN before running."
            ) from exc
        raise


# ======================= dataset helpers =======================
class InferenceSlideLoader:
    def __init__(self, dfs: pd.DataFrame):
        self.dfs = dfs.reset_index(drop=True)

    def __getitem__(self, index):
        row = self.dfs.iloc[index]
        return row.slide

    def __len__(self):
        return len(self.dfs)


class InferenceTileDataset(data.Dataset):
    def __init__(
        self,
        dft: pd.DataFrame,
        dfs: pd.DataFrame,
        tilesize: int,
        slide_handles=None,
    ):
        self.dft = dft
        self.tilesize = tilesize
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.485, 0.456, 0.406),
                    (0.229, 0.224, 0.225),
                ),
            ]
        )
        self.curr = pd.DataFrame()
        self.slide_handles = {}
        self._owned_handles = set()
        for _, row in dfs.iterrows():
            sid = str(row.slide)
            if slide_handles is not None and sid in slide_handles:
                self.slide_handles[sid] = slide_handles[sid]
            else:
                self.slide_handles[sid] = openslide.OpenSlide(row.slide_path)
                self._owned_handles.add(sid)

    def set_slide(self, slide):
        self.curr = self.dft[self.dft.slide == slide].reset_index(drop=True)
        self.curr_slide = str(slide)

    def __getitem__(self, index):
        row = self.curr.iloc[index]
        slide = self.slide_handles[str(row.slide)]
        mult = float(row.mult)
        level = int(row.level)
        size = int(round(self.tilesize * mult))
        region = slide.read_region((int(row.x), int(row.y)), level, (size, size)).convert("RGB")
        if mult != 1:
            region = region.resize((self.tilesize, self.tilesize), Image.LANCZOS)
        return self.transform(region)

    def __len__(self):
        return len(self.curr)

    def close(self):
        for sid in self._owned_handles:
            try:
                self.slide_handles[sid].close()
            except Exception:
                pass
        self._owned_handles.clear()

    def __del__(self):
        self.close()


def get_test_dataset(dfs, dft, tilesize=224, slide_handles=None):
    tile_dataset = InferenceTileDataset(dft, dfs, tilesize, slide_handles=slide_handles)
    slide_loader = InferenceSlideLoader(dfs)
    return tile_dataset, slide_loader


# ======================= tissue extraction helpers =======================
def slide_base_mpp(slide):
    value = slide.properties.get(openslide.PROPERTY_NAME_MPP_X)
    if value is None:
        raise ValueError("Slide missing MPP metadata")
    return float(value)


def find_level(slide, mpp, patchsize=224, base_mpp=None):
    if base_mpp is None:
        base_mpp = ISYNTAX_MPP
    downsample = mpp / base_mpp
    level = None
    mult = None
    for i in range(slide.level_count)[::-1]:
        if (
            abs(downsample / slide.level_downsamples[i] * patchsize - patchsize)
            < MAX_PIXEL_DIFFERENCE * patchsize
            or downsample > slide.level_downsamples[i]
        ):
            level = i
            mult = downsample / slide.level_downsamples[level]
            break
    if level is None:
        raise RuntimeError(f"Requested resolution ({mpp} mpp) is too high for slide")
    mult = np.round(mult * patchsize) / patchsize
    if abs(mult * patchsize - patchsize) < MAX_PIXEL_DIFFERENCE * patchsize:
        mult = 1.0
    return level, mult


def image2array(img):
    if img.__class__.__name__ == "Image":
        if img.mode == "RGB":
            img = np.array(img)
            r, g, b = np.rollaxis(img, axis=-1)
            img = np.stack([r, g, b], axis=-1)
        elif img.mode == "RGBA":
            img = np.array(img)
            r, g, b, a = np.rollaxis(img, axis=-1)
            img = np.stack([r, g, b], axis=-1)
        else:
            raise ValueError("Unsupported image mode")
    img = np.uint8(img)
    return img


def detect_marker(thumb, mult):
    ksize = int(max(1, mult))
    img = cv2.GaussianBlur(thumb, (5, 5), 0)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    black_marker = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 255, 125]))
    blue_marker = cv2.inRange(hsv, np.array([90, 30, 30]), np.array([130, 255, 255]))
    green_marker = cv2.inRange(hsv, np.array([40, 30, 30]), np.array([90, 255, 255]))
    mask = cv2.bitwise_or(cv2.bitwise_or(black_marker, blue_marker), green_marker)
    mask = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize)))
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize * 3, ksize * 3)))
    return mask if np.count_nonzero(mask) > 0 else None


def filter_regions(img, min_size):
    labeled, n = label(img, return_num=True)
    for i in range(1, n + 1):
        if labeled[labeled == i].size < min_size:
            labeled[labeled == i] = 0
    return labeled


def add(overlap):
    return np.linspace(0, 1, overlap + 1)[1:-1]


def add2offset(img, slide, patch_size, mpp, maxmpp):
    size_x = img.shape[1]
    size_y = img.shape[0]
    offset_x = np.floor(
        (slide.dimensions[0] * 1.0 / (patch_size * mpp / maxmpp) - size_x)
        * (patch_size * mpp / maxmpp)
    )
    offset_y = np.floor(
        (slide.dimensions[1] * 1.0 / (patch_size * mpp / maxmpp) - size_y)
        * (patch_size * mpp / maxmpp)
    )
    add_x = np.linspace(0, offset_x, size_x).astype(int)
    add_y = np.linspace(0, offset_y, size_y).astype(int)
    return add_x, add_y


def addoverlap(w, grid, overlap, patch_size, mpp, maxmpp, img, offset=0):
    o = (add(overlap) * (patch_size * mpp / maxmpp)).astype(int)
    ox, oy = np.meshgrid(o, o)
    connx = np.zeros(img.shape).astype(bool)
    conny = np.zeros(img.shape).astype(bool)
    connd = np.zeros(img.shape).astype(bool)
    connu = np.zeros(img.shape).astype(bool)
    connx[:, :-1] = img[:, 1:]
    conny[:-1, :] = img[1:, :]
    connd[:-1, :-1] = img[1:, 1:]
    connu[1:, :-1] = img[:-1, 1:] & (~img[1:, 1:] | ~img[:-1, :-1])
    connx = connx[w]
    conny = conny[w]
    connd = connd[w]
    connu = connu[w]
    extra = []
    for i, (x, y) in enumerate(grid):
        if connx[i]:
            extra.extend(zip(o + x - offset, np.repeat(y, overlap - 1) - offset))
        if conny[i]:
            extra.extend(zip(np.repeat(x, overlap - 1) - offset, o + y - offset))
        if connd[i]:
            extra.extend(zip(ox.flatten() + x - offset, oy.flatten() + y - offset))
        if connu[i]:
            extra.extend(zip(x + ox.flatten() - offset, y - oy.flatten() - offset))
    return extra


def threshold(slide, size, mpp, base_mpp, mult=1):
    w = int(np.round(slide.dimensions[0] * 1.0 / (size * mpp / base_mpp))) * mult
    h = int(np.round(slide.dimensions[1] * 1.0 / (size * mpp / base_mpp))) * mult
    thumbnail = slide.get_thumbnail((w, h))
    thumbnail = thumbnail.resize((w, h))
    img_c = image2array(thumbnail)
    std = np.std(img_c, axis=-1)
    img_g = cv2.cvtColor(img_c, cv2.COLOR_RGB2GRAY)
    marker = detect_marker(img_c, base_mpp / mpp * mult)
    img_g = cv2.GaussianBlur(img_g, (5, 5), 0)
    if marker is not None:
        masked = np.ma.masked_array(img_g, (marker > 0) | (img_g == 255))
        t = threshold_otsu(masked.compressed())
        img_g = cv2.threshold(img_g, t, 255, cv2.THRESH_BINARY)[1]
        img_g = cv2.subtract(~img_g, marker)
    else:
        masked = np.ma.masked_array(img_g, img_g == 255)
        t = threshold_otsu(masked.compressed())
        img_g = cv2.threshold(img_g, t, 255, cv2.THRESH_BINARY)[1]
        img_g = 255 - img_g
    img_g[std < 5] = 0
    if mult > 1:
        img_g = img_g.reshape(h // mult, mult, w // mult, mult).max(axis=(1, 3))
    return img_g, t


def make_sample_grid(
    slide,
    patch_size=224,
    mpp=0.5,
    min_cc_size=10,
    max_ratio_size=10,
    dilate=False,
    erode=False,
    prune=False,
    overlap=1,
    maxn=None,
    bmp=None,
    oversample=False,
    mult=1,
    centerpixel=False,
    base_mpp=None,
):
    if base_mpp is None:
        base_mpp = ISYNTAX_MPP
    if oversample:
        img, th = threshold(slide, patch_size, base_mpp, base_mpp, mult)
    else:
        img, th = threshold(slide, patch_size, mpp, base_mpp, mult)
    img = filter_regions(img, min_cc_size)
    img[img > 0] = 1
    if bmp:
        bmplab = Image.open(bmp)
        thumbx, thumby = img.shape
        bmplab = bmplab.resize((thumby, thumbx), Image.LANCZOS)
        bmplab = np.array(bmplab)
        bmplab[bmplab > 0] = 1
        img = np.logical_and(img, bmplab)
    if erode:
        img = binary_erosion(img)
    if dilate:
        img = binary_dilation(img)
    if oversample:
        add_x, add_y = add2offset(img, slide, patch_size, base_mpp, base_mpp)
    else:
        add_x, add_y = add2offset(img, slide, patch_size, mpp, base_mpp)
    w = np.where(img > 0)
    if oversample:
        offset = int(0.5 * patch_size * ((mpp / base_mpp) - 1))
        grid = list(
            zip(
                (w[1] * (patch_size) + add_x[w[1]] - offset).astype(int),
                (w[0] * (patch_size) + add_y[w[0]] - offset).astype(int),
            )
        )
    else:
        grid = list(
            zip(
                (w[1] * (patch_size * mpp / base_mpp) + add_x[w[1]]).astype(int),
                (w[0] * (patch_size * mpp / base_mpp) + add_y[w[0]]).astype(int),
            )
        )
    if overlap > 1:
        if oversample:
            extra = addoverlap(
                w,
                grid,
                overlap,
                patch_size,
                base_mpp,
                base_mpp,
                img,
                offset=offset,
            )
        else:
            extra = addoverlap(
                w,
                grid,
                overlap,
                patch_size,
                mpp,
                base_mpp,
                img,
            )
        grid.extend(extra)
    if centerpixel:
        offset_center = int(mpp / base_mpp * patch_size // 2)
        grid = [(x + offset_center, y + offset_center) for x, y in grid]
    if prune:
        level, mult = find_level(slide, mpp, base_mpp=base_mpp)
        psize = int(patch_size * mult)
        truegrid = []
        for tup in grid:
            reg = slide.read_region(tup, level, (psize, psize))
            if mult != 1:
                reg = reg.resize((224, 224), Image.BILINEAR)
            reg = image2array(reg)
            if _is_sample(reg, th / 255, 0.2, 0.4, 0.5):
                truegrid.append(tup)
    else:
        truegrid = grid
    if maxn:
        truegrid = random.sample(truegrid, min(maxn, len(truegrid)))
    return truegrid


def _is_sample(img, threshold=0.9, ratio_center=0.1, whole_cutoff=0.5, center_cutoff=0.9):
    nrows, ncols = img.shape[:2]
    _, timg = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), int(255 * threshold), 1, cv2.THRESH_BINARY_INV)
    kernel = np.ones((5, 5), np.uint8)
    cimg = cv2.morphologyEx(timg, cv2.MORPH_CLOSE, kernel)
    crow = int(round(nrows / 2))
    ccol = int(round(ncols / 2))
    drow = int(round(nrows * ratio_center / 2))
    dcol = int(round(ncols * ratio_center / 2))
    centerw = cimg[crow - drow : crow + drow, ccol - dcol : ccol + dcol]
    if (np.count_nonzero(cimg) < nrows * ncols * whole_cutoff) and (
        np.count_nonzero(centerw) < 4 * drow * dcol * center_cutoff
    ):
        return False
    return True


def save_grid_thumbnail(slide, grid, slide_id, level, mult, thumb_dir, tile_size=224, max_dim=1536):
    os.makedirs(thumb_dir, exist_ok=True)
    width, height = slide.dimensions
    scale = max(width, height) / max_dim if max(width, height) > max_dim else 1.0
    thumb_w = int(round(width / scale))
    thumb_h = int(round(height / scale))
    base_thumb = slide.get_thumbnail((thumb_w, thumb_h)).convert("RGB")

    base_filename = f"{slide_id}_thumb.png"
    base_path = os.path.join(thumb_dir, base_filename)
    base_thumb.save(base_path, "PNG")

    overlay_thumb = base_thumb.copy()
    draw = ImageDraw.Draw(overlay_thumb)

    ratio_x = overlay_thumb.width / width
    ratio_y = overlay_thumb.height / height
    level_downsample = float(slide.level_downsamples[level])
    patch_size_level = int(round(tile_size * mult))
    patch_size_base = max(1, int(round(patch_size_level * level_downsample)))

    for x, y in grid:
        x0 = x * ratio_x
        y0 = y * ratio_y
        x1 = (x + patch_size_base) * ratio_x
        y1 = (y + patch_size_base) * ratio_y
        draw.rectangle([x0, y0, x1, y1], outline="red", width=2)

    filename = f"{slide_id}_grid.png"
    grid_path = os.path.join(thumb_dir, filename)
    overlay_thumb.save(grid_path, "PNG")
    return grid_path, base_path


# ======================= inference helpers =======================
@torch.inference_mode()
def get_embedding(loader, model, device, desc=None):
    model.eval()
    ndim = getattr(model, "ndim")
    out = torch.empty((len(loader.dataset), ndim), dtype=torch.float32, device="cpu")
    i0 = 0
    use_cuda = device.type == "cuda"
    iterator = loader
    progress = None
    if desc is not None:
        try:
            total_batches = len(loader)
        except TypeError:
            total_batches = None
        progress = tqdm(loader, total=total_batches, desc=desc, leave=False)
        iterator = progress
    for img in iterator:
        if use_cuda:
            img = img.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=True):
                h = model(img)
        else:
            h = model(img)
        bs = h.size(0)
        out[i0 : i0 + bs].copy_(h.detach().cpu())
        i0 += bs
        del h
        if use_cuda:
            del img
    if progress is not None:
        progress.close()
    return out


@torch.inference_mode()
def infer_one_slide(dfs, dft, tile_model, slide_model, device, slide_handles, batch_size=1, workers=0):
    tile_dataset, slide_loader = get_test_dataset(
        dfs=dfs,
        dft=dft,
        tilesize=224,
        slide_handles=slide_handles,
    )

    pin = device.type == "cuda"
    workers_eff = max(0, int(workers))
    if pin and workers_eff == 0:
        workers_eff = 2

    dl_kwargs = dict(
        batch_size=max(1, int(batch_size)),
        shuffle=False,
        num_workers=workers_eff,
        pin_memory=pin,
        persistent_workers=(workers_eff > 0 and pin),
    )
    if workers_eff > 0:
        dl_kwargs["prefetch_factor"] = 2

    val_tile_loader = torch.utils.data.DataLoader(tile_dataset, **dl_kwargs)

    try:
        for filename in slide_loader:
            tile_dataset.set_slide(filename)
            h = get_embedding(
                val_tile_loader,
                tile_model,
                device,
                desc=f"{filename} tiles",
            )
            _, _, logits = slide_model(h)
            probs = F.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
            return float(probs[0])
    finally:
        tile_dataset.close()
    return None


# ======================= main =======================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slides_dir", required=True, help="Directory containing .svs files")
    ap.add_argument("--outdir", default="/media/hdd1/chad/EAGLE_cytology/results")
    ap.add_argument("--outname", default="epic_cases_cytology.csv")
    ap.add_argument("--tile_checkpoint", required=True)
    ap.add_argument("--slide_checkpoint", required=True)
    ap.add_argument("--gpu_min_free_gb", type=float, default=2.0)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--recursive", action="store_true", help="Recursively search for slides")
    ap.add_argument("--hf_token", default="None", help="Optional Hugging Face token for gated repo access")
    args = ap.parse_args()

    hf_token = args.hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if hf_token:
        os.environ.setdefault("HF_TOKEN", hf_token)
        os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", hf_token)
        os.environ.setdefault("HUGGINGFACEHUB_API_TOKEN", hf_token)

    slides_root = Path(args.slides_dir).expanduser().resolve()
    if not slides_root.is_dir():
        raise FileNotFoundError(f"slides_dir not found: {slides_root}")

    pattern = "**/*.svs" if args.recursive else "*.svs"
    svs_paths = [p for p in slides_root.glob(pattern) if p.is_file()]
    if not svs_paths:
        raise FileNotFoundError(f"No .svs files found under {slides_root}")

    os.makedirs(args.outdir, exist_ok=True)
    thumb_dir = os.path.join(args.outdir, THUMB_SUBDIR)
    os.makedirs(thumb_dir, exist_ok=True)

    final_out = os.path.join(args.outdir, args.outname)

    print(
        {
            "torch_version": torch.__version__,
            "cuda_compiled": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "visible_devices": torch.cuda.device_count(),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        }
    )
    if torch.cuda.is_available():
        print({f"free_gb_cuda:{i}": round(free_vram_gb(i), 2) for i in range(torch.cuda.device_count())})

    if os.path.exists(final_out) and os.path.getsize(final_out) > 0:
        out_df = pd.read_csv(final_out)
    else:
        out_df = pd.DataFrame()

    score_series = out_df["score"] if "score" in out_df else pd.Series(dtype=float)
    ntiles_series = out_df["n_tiles"] if "n_tiles" in out_df else pd.Series(dtype=float)
    score_num = pd.to_numeric(score_series, errors="coerce")
    ntiles_num = pd.to_numeric(ntiles_series, errors="coerce")
    completed_mask = score_num.notna() & ntiles_num.notna() & (ntiles_num > 0)

    already_slides = set(out_df["slide"].astype(str)) if "slide" in out_df else set()
    completed_paths = (
        set(out_df.loc[completed_mask, "slide_path"].astype(str))
        if "slide_path" in out_df
        else set()
    )

    svs_paths_sorted = sorted(svs_paths, key=lambda p: p.stat().st_mtime, reverse=True)

    tile_model = load_tile_model()
    tile_model.ndim = 1536
    slide_model = GMA(ndim=tile_model.ndim, dropout=True)

    tile_ch = torch.load(args.tile_checkpoint, map_location="cpu")
    slide_ch = torch.load(args.slide_checkpoint, map_location="cpu")
    tile_model.load_state_dict(fix_state_dict(tile_ch["tile_model"]))
    slide_model.load_state_dict(slide_ch["slide_model"])
    slide_model = slide_model.cpu().eval()

    for idx, svs_path in enumerate(svs_paths_sorted, start=1):
        svs_path_str = str(svs_path)
        slide_id = os.path.splitext(os.path.basename(svs_path_str))[0]

        if svs_path_str in completed_paths or slide_id in already_slides:
            print(f"[{idx}/{len(svs_paths_sorted)}] Skip (completed or present): {slide_id} path={svs_path_str}")
            continue

        if not os.path.isfile(svs_path_str):
            print(f"[{idx}/{len(svs_paths_sorted)}] Skip (missing file): {slide_id} path={svs_path_str}")
            continue

        device = pick_device(args.gpu_min_free_gb)
        using_cuda = device.type == "cuda"
        print(f"[{idx}/{len(svs_paths_sorted)}] Device chosen: {device}")
        if using_cuda:
            try:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.benchmark = False
                torch.backends.cudnn.deterministic = True
                torch.set_grad_enabled(False)
                torch.cuda.empty_cache()
            except Exception:
                pass
            tile_model = tile_model.to(device).eval()
        else:
            tile_model = tile_model.cpu().eval()

        slide = None
        try:
            slide = openslide.OpenSlide(svs_path_str)
            base_mpp = slide_base_mpp(slide)
            level, mult = find_level(slide, 0.5, patchsize=224, base_mpp=base_mpp)
            grid = make_sample_grid(
                slide,
                patch_size=224,
                mpp=0.5,
                mult=4,
                base_mpp=base_mpp,
            )
            n_tiles = len(grid)
            thumbnail_path, thumbnail_clean_path = save_grid_thumbnail(
                slide,
                grid,
                slide_id,
                level,
                mult,
                thumb_dir,
                tile_size=224,
            )

            dft = pd.DataFrame(
                [{"x": x, "y": y, "slide": slide_id, "level": level, "mult": mult} for x, y in grid]
            )
            dfs = pd.DataFrame(
                [
                    {
                        "slide": slide_id,
                        "slide_path": svs_path_str,
                        "target": 0,
                        "n_tiles": n_tiles,
                        "level": level,
                    }
                ]
            )

            slide_handles = {slide_id: slide}
            try:
                score = infer_one_slide(
                    dfs,
                    dft,
                    tile_model,
                    slide_model,
                    device,
                    slide_handles,
                    batch_size=args.batch_size,
                    workers=args.workers,
                )
            except RuntimeError as e:
                if "CUDA out of memory" in str(e) or "CUDACachingAllocator" in str(e):
                    print(f"[{idx}] CUDA tight -> retrying on CPU for {slide_id}")
                    tile_model = tile_model.cpu().eval()
                    device = torch.device("cpu")
                    score = infer_one_slide(
                        dfs,
                        dft,
                        tile_model,
                        slide_model,
                        device,
                        slide_handles,
                        batch_size=1,
                        workers=0,
                    )
                else:
                    raise

            if score is None:
                print(f"[{idx}] No score produced for {slide_id}, skipping write.")
                continue

            row = OrderedDict(
                [
                    ("slide", slide_id),
                    ("slide_path", svs_path_str),
                    ("target", 0),
                    ("score", score),
                    ("inference_time", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
                    ("n_tiles", n_tiles),
                    ("level", level),
                    ("thumbnail_path", thumbnail_path),
                    ("thumbnail_clean_path", thumbnail_clean_path),
                ]
            )
            out_row = pd.DataFrame([row])
            append_header = not os.path.exists(final_out) or os.path.getsize(final_out) == 0
            out_row.to_csv(
                final_out,
                mode="a",
                header=append_header,
                index=False,
            )

            print(
                "[{}/{}] Saved {} score={:.6f} n_tiles={} level={}".format(
                    idx,
                    len(svs_paths_sorted),
                    slide_id,
                    score,
                    n_tiles,
                    level,
                )
            )

            already_slides.add(slide_id)
            completed_paths.add(svs_path_str)

        except Exception as e:
            print(f"[{idx}] Error {slide_id}: {e}")
        finally:
            try:
                if slide is not None:
                    slide.close()
            except Exception:
                pass
            for obj_name in ["dft", "dfs", "grid"]:
                if obj_name in locals():
                    try:
                        del locals()[obj_name]
                    except Exception:
                        pass
            gc.collect()
            if torch.cuda.is_available() and using_cuda:
                torch.cuda.empty_cache()

    print("All done.")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning, module="openslide")
    main()
