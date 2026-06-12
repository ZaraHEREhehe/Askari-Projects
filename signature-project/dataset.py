"""
dataset.py
==========
Image-level dataset and batch sampler for online triplet mining.

build_cache(root_dir, cache_dir)
    Preprocess all 2640 images once to disk as uint8 .npy arrays.
    Re-runs skip already-cached files — safe to call every training run.

CEDARImageDataset(cache_dir, writers)
    Returns (image_tensor, writer_id, is_genuine) per image.
    Loads from cache and applies ImageNet-grayscale normalisation.

PKSampler(dataset, P, K_genuine, K_forgery, num_batches, seed)
    Yields P writers x (K_genuine + K_forgery) image indices per batch.
    Re-randomised every __iter__ call so each epoch sees fresh combinations.
"""

import os
import random

import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset, Sampler

from preprocess import preprocess_signature

# ImageNet statistics collapsed to a single grayscale channel.
# Pretrained ResNet-18 expects inputs in this range; skipping this
# normalisation degrades pretrained feature quality from the first forward pass.
IMAGENET_MEAN = 0.449
IMAGENET_STD  = 0.226

TRAIN_WRITERS    = list(range(1,  46))
TEST_WRITERS     = list(range(46, 56))
SAMPLES_EACH     = 24

D2_WRITER_OFFSET = 1000   # dataset2 IDs mapped to 1001–1686 (no clash with CEDAR)
D2_SAMPLES_EACH  = 10     # 10 genuine + 10 forgery per dataset2 writer

# Writers 1-650 are used for training; 651-686 are held out for testing.
# This gives an honest evaluation on data the model has never seen.
D2_TRAIN_WRITERS = list(range(1,   651))   # 650 writers for training
D2_TEST_WRITERS  = list(range(651, 687))   #  36 writers held out for testing


# ---------------------------------------------------------------------------
# Cache builder
# ---------------------------------------------------------------------------

def build_cache(root_dir: str, cache_dir: str) -> None:
    """
    Preprocess every image once and save as a uint8 (224x224) .npy file.

    The preprocessing pipeline (binarise, crop, pad, resize) is expensive
    when run per-sample per-epoch from raw PNG.  Caching moves that cost to
    a one-time step: subsequent epochs do a fast np.load() instead.

    Naming: org_{writer}_{i}.npy  and  forg_{writer}_{i}.npy
    """
    os.makedirs(cache_dir, exist_ok=True)
    org_dir  = os.path.join(root_dir, "full_org")
    forg_dir = os.path.join(root_dir, "full_forg")

    tasks = []
    for wid in range(1, 56):
        for i in range(1, SAMPLES_EACH + 1):
            tasks.append((
                os.path.join(org_dir,   f"original_{wid}_{i}.png"),
                os.path.join(cache_dir, f"org_{wid}_{i}.npy"),
            ))
            tasks.append((
                os.path.join(forg_dir,  f"forgeries_{wid}_{i}.png"),
                os.path.join(cache_dir, f"forg_{wid}_{i}.npy"),
            ))

    total = len(tasks)
    n_new = 0
    for src, dst in tasks:
        if os.path.exists(dst):
            continue
        np.save(dst, preprocess_signature(src))
        n_new += 1
        if n_new % 200 == 0:
            print(f"  Cached {n_new} images …")

    if n_new:
        print(f"  Cached {n_new} new images.")
    print(f"Cache ready — {total} total  ({cache_dir})")


# ---------------------------------------------------------------------------
# Image-level dataset
# ---------------------------------------------------------------------------

class CEDARImageDataset(Dataset):
    """
    One item = one signature image with its metadata.

    Returns
    -------
    image_tensor : float32 [1, 224, 224], ImageNet-normalised
    writer_id    : int
    is_genuine   : int  (1 = genuine, 0 = forgery)
    """

    def __init__(self, cache_dir: str, writers: list, augment: bool = False):
        self.cache_dir = cache_dir
        self.augment   = augment
        self.items     = []          # (npy_path, writer_id, is_genuine)

        for wid in writers:
            for i in range(1, SAMPLES_EACH + 1):
                self.items.append(
                    (os.path.join(cache_dir, f"org_{wid}_{i}.npy"),  wid, 1)
                )
                self.items.append(
                    (os.path.join(cache_dir, f"forg_{wid}_{i}.npy"), wid, 0)
                )

        # Index maps for O(1) random selection inside PKSampler.
        self.genuine_by_writer: dict = {}
        self.forgery_by_writer: dict = {}
        for idx, (_, wid, is_gen) in enumerate(self.items):
            if is_gen:
                self.genuine_by_writer.setdefault(wid, []).append(idx)
            else:
                self.forgery_by_writer.setdefault(wid, []).append(idx)

        # Build augmentation pipeline (training only; None when augment=False).
        # Applied in [0,1] float space BEFORE ImageNet normalisation so that
        # brightness/contrast jitter operates in the natural pixel range.
        self._aug = T.Compose([
            T.RandomRotation(degrees=10, fill=1.0),
            T.RandomAffine(degrees=0, scale=(0.9, 1.1), fill=1.0),
            T.RandomPerspective(distortion_scale=0.1, p=0.5, fill=1.0),
            T.ColorJitter(brightness=0.2, contrast=0.2),
            T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
            T.RandomErasing(p=0.5, scale=(0.02, 0.08), value=1.0),
        ]) if augment else None

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        path, writer_id, is_genuine = self.items[idx]
        arr = np.load(path)                                      # uint8 (224, 224)
        img = torch.from_numpy(arr.astype(np.float32) / 255.0)  # float32 [0, 1]
        img = img.unsqueeze(0)                                   # (1, 224, 224)
        if self._aug is not None:
            img = self._aug(img)
        img = (img - IMAGENET_MEAN) / IMAGENET_STD               # ImageNet normalise
        return img, writer_id, is_genuine


# ---------------------------------------------------------------------------
# Batch sampler
# ---------------------------------------------------------------------------

class PKSampler(Sampler):
    """
    Each batch: P writers x (K_genuine genuine + K_forgery forgery) images.

    A new random state is used each __iter__ call so every epoch draws a
    fresh set of writer/image combinations — the "online" part of online mining.

    Parameters
    ----------
    P           : writers per batch (default 16)
    K_genuine   : genuine images per writer per batch (default 4)
    K_forgery   : forgery images per writer per batch (default 4)
    num_batches : batches per epoch
    seed        : set for a reproducible sanity check; None for normal training
    """

    def __init__(
        self,
        dataset:     CEDARImageDataset,
        P:           int  = 16,
        K_genuine:   int  = 4,
        K_forgery:   int  = 4,
        num_batches: int  = 94,
        seed:        int  = None,
    ):
        self.dataset     = dataset
        self.P           = P
        self.K_genuine   = K_genuine
        self.K_forgery   = K_forgery
        self.num_batches = num_batches
        self.seed        = seed
        self.writers     = list(dataset.genuine_by_writer.keys())

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self):
        rng = random.Random(self.seed)          # fresh RNG → different batches each epoch
        for _ in range(self.num_batches):
            chosen  = rng.sample(self.writers, min(self.P, len(self.writers)))
            indices = []
            for wid in chosen:
                # choices() samples with replacement so K > pool_size is safe
                indices += rng.choices(self.dataset.genuine_by_writer[wid], k=self.K_genuine)
                indices += rng.choices(self.dataset.forgery_by_writer[wid], k=self.K_forgery)
            yield indices


# ---------------------------------------------------------------------------
# Stratified batch sampler  (fixes CEDAR / Dataset2 class imbalance)
# ---------------------------------------------------------------------------

class StratifiedPKSampler(Sampler):
    """
    Samples P_cedar CEDAR writers and P_d2 Dataset2 writers every batch.

    Without this, a plain random sampler over 695 writers gives only ~0.8
    CEDAR writers per batch (14:1 skew toward Dataset2).  Fixing this to
    a 4:8 ratio gives CEDAR ~5x more exposure than its natural proportion,
    so the model learns CEDAR patterns properly without losing Dataset2
    diversity.

    Writers are identified by ID: CEDAR IDs < D2_WRITER_OFFSET, Dataset2 IDs >= offset.
    """

    def __init__(
        self,
        dataset:     CombinedDataset,
        P_cedar:     int = 4,
        P_d2:        int = 8,
        K_genuine:   int = 4,
        K_forgery:   int = 4,
        num_batches: int = 125,
        seed:        int = None,
    ):
        self.dataset     = dataset
        self.P_cedar     = P_cedar
        self.P_d2        = P_d2
        self.K_genuine   = K_genuine
        self.K_forgery   = K_forgery
        self.num_batches = num_batches
        self.seed        = seed

        all_writers = list(dataset.genuine_by_writer.keys())
        self.cedar_writers = [w for w in all_writers if w <  D2_WRITER_OFFSET]
        self.d2_writers    = [w for w in all_writers if w >= D2_WRITER_OFFSET]

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self):
        rng = random.Random(self.seed)
        for _ in range(self.num_batches):
            cedar = rng.sample(self.cedar_writers, min(self.P_cedar, len(self.cedar_writers)))
            d2    = rng.sample(self.d2_writers,    min(self.P_d2,    len(self.d2_writers)))
            indices = []
            for wid in cedar + d2:
                indices += rng.choices(self.dataset.genuine_by_writer[wid], k=self.K_genuine)
                indices += rng.choices(self.dataset.forgery_by_writer[wid], k=self.K_forgery)
            yield indices


# ---------------------------------------------------------------------------
# Dataset2 cache builder
# ---------------------------------------------------------------------------

def build_cache_dataset2(dataset2_dir: str, cache_dir: str) -> None:
    """
    Cache dataset2 images (686 writers, ~10 genuine + 10 forgery each).
    Folder layout: dataset2/{NNN}/ for genuine, dataset2/{NNN}_forg/ for forgeries.
    Saved as d2_org_{wid}_{i}.npy and d2_forg_{wid}_{i}.npy.
    """
    os.makedirs(cache_dir, exist_ok=True)

    writer_dirs = sorted(
        d for d in os.listdir(dataset2_dir)
        if os.path.isdir(os.path.join(dataset2_dir, d)) and not d.endswith("_forg")
    )

    total = 0
    n_new = 0
    for wname in writer_dirs:
        wid      = int(wname)
        org_dir  = os.path.join(dataset2_dir, wname)
        forg_dir = os.path.join(dataset2_dir, wname + "_forg")

        org_files  = sorted(f for f in os.listdir(org_dir)  if f.lower().endswith(".jpg"))
        forg_files = sorted(f for f in os.listdir(forg_dir) if f.lower().endswith(".jpg"))

        for i, fname in enumerate(org_files, 1):
            dst = os.path.join(cache_dir, f"d2_org_{wid}_{i}.npy")
            total += 1
            if not os.path.exists(dst):
                np.save(dst, preprocess_signature(os.path.join(org_dir, fname)))
                n_new += 1
                if n_new % 500 == 0:
                    print(f"  Cached {n_new} dataset2 images …")

        for i, fname in enumerate(forg_files, 1):
            dst = os.path.join(cache_dir, f"d2_forg_{wid}_{i}.npy")
            total += 1
            if not os.path.exists(dst):
                np.save(dst, preprocess_signature(os.path.join(forg_dir, fname)))
                n_new += 1
                if n_new % 500 == 0:
                    print(f"  Cached {n_new} dataset2 images …")

    if n_new:
        print(f"  Cached {n_new} new dataset2 images.")
    print(f"Dataset2 cache ready — {total} total  ({cache_dir})")


# ---------------------------------------------------------------------------
# Combined dataset (CEDAR + Dataset2)
# ---------------------------------------------------------------------------

class CombinedDataset(Dataset):
    """
    Merges CEDAR training writers and dataset2 writers into one dataset.
    Dataset2 writer IDs are offset by D2_WRITER_OFFSET to avoid collision
    with CEDAR IDs.  Drop-in replacement for CEDARImageDataset.
    """

    def __init__(
        self,
        cache_dir:     str,
        cedar_writers: list,
        d2_writers:    list,
        augment:       bool = False,
    ):
        self.cache_dir = cache_dir
        self.augment   = augment
        self.items     = []

        # CEDAR items
        for wid in cedar_writers:
            for i in range(1, SAMPLES_EACH + 1):
                self.items.append(
                    (os.path.join(cache_dir, f"org_{wid}_{i}.npy"),  wid, 1)
                )
                self.items.append(
                    (os.path.join(cache_dir, f"forg_{wid}_{i}.npy"), wid, 0)
                )

        # Dataset2 items (writer IDs offset to avoid clash with CEDAR)
        for wid in d2_writers:
            mapped = wid + D2_WRITER_OFFSET
            for i in range(1, D2_SAMPLES_EACH + 1):
                org_p  = os.path.join(cache_dir, f"d2_org_{wid}_{i}.npy")
                forg_p = os.path.join(cache_dir, f"d2_forg_{wid}_{i}.npy")
                if os.path.exists(org_p):
                    self.items.append((org_p,  mapped, 1))
                if os.path.exists(forg_p):
                    self.items.append((forg_p, mapped, 0))

        # Index maps for PKSampler
        self.genuine_by_writer: dict = {}
        self.forgery_by_writer: dict = {}
        for idx, (_, wid, is_gen) in enumerate(self.items):
            if is_gen:
                self.genuine_by_writer.setdefault(wid, []).append(idx)
            else:
                self.forgery_by_writer.setdefault(wid, []).append(idx)

        self._aug = T.Compose([
            T.RandomRotation(degrees=10, fill=1.0),
            T.RandomAffine(degrees=0, scale=(0.9, 1.1), fill=1.0),
            T.RandomPerspective(distortion_scale=0.1, p=0.5, fill=1.0),
            T.ColorJitter(brightness=0.2, contrast=0.2),
            T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
            T.RandomErasing(p=0.5, scale=(0.02, 0.08), value=1.0),
        ]) if augment else None

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        path, writer_id, is_genuine = self.items[idx]
        arr = np.load(path)
        img = torch.from_numpy(arr.astype(np.float32) / 255.0)
        img = img.unsqueeze(0)
        if self._aug is not None:
            img = self._aug(img)
        img = (img - IMAGENET_MEAN) / IMAGENET_STD
        return img, writer_id, is_genuine
