"""
evaluate.py
===========
Tests the trained model on two independent held-out sets:

  CEDAR     — writers 46-55   (10 writers, never seen during training)
  Dataset2  — writers 651-686 (36 writers, never seen during training)

Both sets were excluded from training so results are honest.

Primary banking metric: FAR @ 0.5%
  How many forged cheques slip through at the strictest threshold.
  Lower = better. EER is secondary.

Run:
  python evaluate.py                      # uses checkpoints/best.pt
  python evaluate.py --checkpoint <path>
"""

import os
import itertools
import argparse

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model   import SiameseNetwork
from dataset import (
    IMAGENET_MEAN, IMAGENET_STD,
    TEST_WRITERS,    SAMPLES_EACH,
    D2_TEST_WRITERS, D2_SAMPLES_EACH,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

EMBEDDING_DIM  = 512
BATCH_SIZE     = 64
CHECKPOINT_DIR = "checkpoints"
PLOTS_DIR      = "plots"
DEFAULT_CKPT   = os.path.join(CHECKPOINT_DIR, "best.pt")

# FAR operating points to report — 0.5% is the primary banking metric
FAR_TARGETS = [0.005, 0.01, 0.02, 0.05]


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(checkpoint_path: str, device: torch.device) -> SiameseNetwork:
    """Load trained weights into the model. Handles both checkpoint formats."""
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            "Run train.py first to generate it."
        )

    payload = torch.load(checkpoint_path, map_location=device)

    # train.py saves two formats:
    #   best.pt / final.pt  → just the weights dict
    #   epoch_NN.pt         → full dict with "model_state_dict" key
    if isinstance(payload, dict) and "model_state_dict" in payload:
        state_dict = payload["model_state_dict"]
        print(f"Loaded epoch-{payload.get('epoch', '?')} checkpoint  "
              f"(train loss: {payload.get('loss', float('nan')):.6f})")
    else:
        state_dict = payload
        print(f"Loaded weights from {checkpoint_path}")

    model = SiameseNetwork(embedding_dim=EMBEDDING_DIM)
    model.load_state_dict(state_dict)
    model.eval().to(device)
    return model


# ---------------------------------------------------------------------------
# Pair builders — load from cache (.npy) for speed and consistency
# ---------------------------------------------------------------------------

def build_cedar_test_pairs(cache_dir: str):
    """
    Build all evaluation pairs for CEDAR writers 46-55.
    Loads from cached .npy files (faster than re-preprocessing raw PNGs).

    Returns (genuine_pairs, forgery_pairs)
      genuine_pairs : C(24,2)=276 per writer × 10 = 2,760 total
      forgery_pairs : 24×24=576 per writer × 10   = 5,760 total
    """
    genuine_pairs, forgery_pairs = [], []

    for wid in TEST_WRITERS:
        org_paths = [
            os.path.join(cache_dir, f"org_{wid}_{i}.npy")
            for i in range(1, SAMPLES_EACH + 1)
        ]
        forg_paths = [
            os.path.join(cache_dir, f"forg_{wid}_{i}.npy")
            for i in range(1, SAMPLES_EACH + 1)
        ]

        # Only keep files that were actually cached
        org_paths  = [p for p in org_paths  if os.path.exists(p)]
        forg_paths = [p for p in forg_paths if os.path.exists(p)]

        # All unordered pairs of genuine signatures from the same writer
        genuine_pairs.extend(itertools.combinations(org_paths, 2))
        # Every genuine paired with every forgery from the same writer
        forgery_pairs.extend(
            (g, f) for g in org_paths for f in forg_paths
        )

    return genuine_pairs, forgery_pairs


def build_d2_test_pairs(cache_dir: str):
    """
    Build all evaluation pairs for Dataset2 writers 651-686.
    Loads from cached .npy files.

    Returns (genuine_pairs, forgery_pairs)
      ~C(10,2)=45 genuine + ~10×10=100 forgery pairs per writer × 36 writers
    """
    genuine_pairs, forgery_pairs = [], []

    for wid in D2_TEST_WRITERS:
        org_paths = [
            os.path.join(cache_dir, f"d2_org_{wid}_{i}.npy")
            for i in range(1, D2_SAMPLES_EACH + 1)
        ]
        forg_paths = [
            os.path.join(cache_dir, f"d2_forg_{wid}_{i}.npy")
            for i in range(1, D2_SAMPLES_EACH + 1)
        ]

        org_paths  = [p for p in org_paths  if os.path.exists(p)]
        forg_paths = [p for p in forg_paths if os.path.exists(p)]

        # Skip writers with too few samples to form meaningful pairs
        if len(org_paths) < 2 or not forg_paths:
            continue

        genuine_pairs.extend(itertools.combinations(org_paths, 2))
        forgery_pairs.extend(
            (g, f) for g in org_paths for f in forg_paths
        )

    return genuine_pairs, forgery_pairs


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def compute_scores(
    model:      SiameseNetwork,
    pairs:      list,
    device:     torch.device,
    batch_size: int = 64,
    label:      str = "",
) -> np.ndarray:
    """
    Run all pairs through the model and return cosine similarity scores.

    Loads images from .npy cache files — no preprocessing overhead.
    Higher score = more similar (genuine pair expected near 1.0,
    forgery pair expected near 0 or below).
    """
    scores  = []
    n_pairs = len(pairs)

    with torch.no_grad():
        for start in range(0, n_pairs, batch_size):
            batch = pairs[start : start + batch_size]

            imgs_a, imgs_b = [], []
            for path_a, path_b in batch:
                for path, bucket in ((path_a, imgs_a), (path_b, imgs_b)):
                    # Load uint8 cache → float [0,1] → ImageNet normalise
                    arr = np.load(path).astype(np.float32) / 255.0
                    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
                    bucket.append(torch.from_numpy(arr).unsqueeze(0))

            emb_a, emb_b = model(
                torch.stack(imgs_a).to(device),
                torch.stack(imgs_b).to(device),
            )
            scores.extend(
                F.cosine_similarity(emb_a, emb_b, dim=1).cpu().numpy()
            )

            done = min(start + batch_size, n_pairs)
            print(f"  {label:10s}  {done:>5}/{n_pairs} pairs", end="\r")

    print()
    return np.array(scores, dtype=np.float32)


# ---------------------------------------------------------------------------
# EER computation
# ---------------------------------------------------------------------------

def compute_eer(
    genuine_scores: np.ndarray,
    forgery_scores: np.ndarray,
) -> tuple:
    """
    Compute Equal Error Rate — the threshold where FAR equals FRR.

    FAR(τ) = fraction of forgery pairs with score ≥ τ  (model fooled)
    FRR(τ) = fraction of genuine pairs with score < τ  (genuine rejected)

    Returns (eer, eer_threshold, thresholds, fars, frrs).
    """
    thresholds  = np.sort(np.unique(
        np.concatenate([genuine_scores, forgery_scores])
    ))
    forg_sorted = np.sort(forgery_scores)
    gen_sorted  = np.sort(genuine_scores)
    n_forg      = len(forgery_scores)
    n_gen       = len(genuine_scores)

    # Vectorised FAR / FRR at every threshold
    fars = (n_forg - np.searchsorted(forg_sorted, thresholds, side="left")) / n_forg
    frrs =           np.searchsorted(gen_sorted,  thresholds, side="left")  / n_gen

    # Find where FAR and FRR cross, then interpolate for a precise EER
    sign_changes = np.where(np.diff(np.sign(fars - frrs)))[0]
    if len(sign_changes) > 0:
        i     = sign_changes[0]
        d_far = fars[i + 1] - fars[i]
        d_frr = frrs[i + 1] - frrs[i]
        denom = d_far - d_frr
        s     = (frrs[i] - fars[i]) / denom if abs(denom) > 1e-12 else 0.5
        eer_threshold = float(thresholds[i] + s * (thresholds[i + 1] - thresholds[i]))
        eer           = float(fars[i] + s * d_far)
    else:
        idx           = int(np.argmin(np.abs(fars - frrs)))
        eer_threshold = float(thresholds[idx])
        eer           = float((fars[idx] + frrs[idx]) / 2)

    return eer, eer_threshold, thresholds, fars, frrs


# ---------------------------------------------------------------------------
# Operating-point table
# ---------------------------------------------------------------------------

def compute_operating_points(
    thresholds:  np.ndarray,
    fars:        np.ndarray,
    frrs:        np.ndarray,
    far_targets: list,
) -> list:
    """
    For each FAR budget find the decision threshold and resulting FRR.

    We pick the LOWEST threshold that still keeps FAR ≤ budget.
    Lowest = least restrictive gate, so FRR is minimised at that FAR level.
    This is correct for banking: set the forgery bar, then see how many
    genuine cheques get wrongly rejected.
    """
    rows = []
    for target in far_targets:
        valid = np.where(fars <= target)[0]
        if len(valid) == 0:
            rows.append({"far_target": target,
                         "threshold":  None,
                         "actual_far": None,
                         "frr":        None})
        else:
            i = valid[0]
            rows.append({"far_target": target,
                         "threshold":  float(thresholds[i]),
                         "actual_far": float(fars[i]),
                         "frr":        float(frrs[i])})
    return rows


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _plot_distributions(
    gen_scores:  np.ndarray,
    forg_scores: np.ndarray,
    eer_thr:     float,
    far1_thr:    float,
    plots_dir:   str,
    name:        str,
) -> None:
    """Overlapping histogram: genuine vs forgery similarity scores."""
    fig, ax = plt.subplots(figsize=(9, 5))
    bins = np.linspace(-1.0, 1.0, 61)

    ax.hist(gen_scores,  bins=bins, color="steelblue", alpha=0.55,
            label=f"Genuine-Genuine  (n={len(gen_scores):,})")
    ax.hist(forg_scores, bins=bins, color="tomato",    alpha=0.55,
            label=f"Genuine-Forgery  (n={len(forg_scores):,})")

    ax.axvline(eer_thr, color="black", linestyle="--", linewidth=1.5,
               label=f"EER threshold = {eer_thr:.3f}")
    if far1_thr is not None:
        ax.axvline(far1_thr, color="red", linestyle="--", linewidth=1.5,
                   label=f"FAR=1% threshold = {far1_thr:.3f}")

    ax.set_xlabel("Cosine Similarity",  fontsize=12)
    ax.set_ylabel("Number of Pairs",    fontsize=12)
    ax.set_title(f"Score Distributions — {name}", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.4)

    slug = name.lower().replace(" ", "_")
    path = os.path.join(plots_dir, f"{slug}_score_dist.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


def _plot_far_frr(
    thresholds: np.ndarray,
    fars:       np.ndarray,
    frrs:       np.ndarray,
    eer:        float,
    eer_thr:    float,
    plots_dir:  str,
    name:       str,
) -> None:
    """FAR and FRR vs threshold with EER operating point marked."""
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(thresholds, fars, color="tomato",    linewidth=2, label="FAR (forgeries accepted)")
    ax.plot(thresholds, frrs, color="steelblue", linewidth=2, label="FRR (genuine rejected)")
    ax.scatter([eer_thr], [eer], color="black", zorder=5, s=80,
               label=f"EER = {eer * 100:.2f}%  @ {eer_thr:.3f}")

    # Dotted cross-hairs to the axes
    ax.axvline(eer_thr, color="black", linestyle=":", linewidth=1)
    ax.axhline(eer,     color="black", linestyle=":", linewidth=1)

    ax.set_xlabel("Decision Threshold (cosine similarity)", fontsize=12)
    ax.set_ylabel("Error Rate",                             fontsize=12)
    ax.set_title(f"FAR / FRR Curve — {name}",              fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-0.02, 1.02)

    slug = name.lower().replace(" ", "_")
    path = os.path.join(plots_dir, f"{slug}_far_frr.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


# ---------------------------------------------------------------------------
# Single-dataset evaluation helper
# ---------------------------------------------------------------------------

def _evaluate_one(
    name:          str,
    genuine_pairs: list,
    forgery_pairs: list,
    model:         SiameseNetwork,
    device:        torch.device,
    plots_dir:     str,
) -> dict:
    """
    Score all pairs, compute EER and operating points, save plots.
    Returns a results dict consumed by the summary table.
    """
    print(f"\n── {name} {'─' * (50 - len(name))}")
    print(f"  Genuine-genuine : {len(genuine_pairs):,}")
    print(f"  Genuine-forgery : {len(forgery_pairs):,}")

    gen_scores  = compute_scores(model, genuine_pairs, device, BATCH_SIZE, "genuine")
    forg_scores = compute_scores(model, forgery_pairs, device, BATCH_SIZE, "forgery")

    print(f"\n  Score stats:")
    print(f"    Genuine  mean={gen_scores.mean():.4f}  std={gen_scores.std():.4f}")
    print(f"    Forgery  mean={forg_scores.mean():.4f}  std={forg_scores.std():.4f}")

    eer, eer_thr, thresholds, fars, frrs = compute_eer(gen_scores, forg_scores)
    ops = compute_operating_points(thresholds, fars, frrs, FAR_TARGETS)

    print(f"\n  EER : {eer * 100:.2f}%  @ threshold {eer_thr:.4f}")

    # FAR=1% threshold used as a second marker on the histogram
    far1_row = next((r for r in ops if r["far_target"] == 0.01), None)
    far1_thr = far1_row["threshold"] if far1_row else None

    print("  Saving plots …")
    _plot_distributions(gen_scores, forg_scores, eer_thr, far1_thr, plots_dir, name)
    _plot_far_frr(thresholds, fars, frrs, eer, eer_thr, plots_dir, name)

    return {"name": name, "eer": eer, "eer_thr": eer_thr, "ops": ops}


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def _print_summary(results_list: list) -> None:
    """
    Side-by-side comparison of all evaluated datasets.
    FAR≤0.5% row is the primary banking metric — forgeries accepted
    at the strictest threshold we report.
    """
    W = 18   # column width

    print("\n" + "=" * (24 + W * len(results_list)))
    print("  EVALUATION SUMMARY  —  lower is better")
    print("=" * (24 + W * len(results_list)))

    header = f"  {'Metric':<22}"
    for r in results_list:
        header += f"  {r['name']:>{W - 2}}"
    print(header)
    print("  " + "-" * (20 + W * len(results_list)))

    # EER
    row = f"  {'EER':<22}"
    for r in results_list:
        row += f"  {r['eer'] * 100:>{W - 2}.2f}%"
    print(row)

    # Operating-point rows
    for target in FAR_TARGETS:
        label = f"FAR≤{target * 100:.1f}%  →  FRR"
        row   = f"  {label:<22}"
        for r in results_list:
            op = next((o for o in r["ops"] if o["far_target"] == target), None)
            if op and op["frr"] is not None:
                row += f"  {op['frr'] * 100:>{W - 2}.2f}%"
            else:
                row += f"  {'N/A':>{W - 2}}"
        print(row)

    print("=" * (24 + W * len(results_list)))
    print("\n  ★  Primary banking metric: FAR≤0.5% → FRR")
    print("     = genuine cheques rejected when forgery gate is at 0.5% leakage\n")


# ---------------------------------------------------------------------------
# Main evaluation pipeline
# ---------------------------------------------------------------------------

def evaluate(checkpoint_path: str = DEFAULT_CKPT) -> list:
    """
    Full evaluation:
      1. Load model
      2. Score CEDAR test writers 46-55
      3. Score Dataset2 test writers 651-686
      4. Print side-by-side summary with banking operating points

    Returns the list of result dicts (useful when called from a notebook).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device     : {device}")
    print(f"Checkpoint : {checkpoint_path}")
    print("-" * 50)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    cache_dir  = os.path.join(script_dir, "cache")
    plots_dir  = os.path.join(script_dir, PLOTS_DIR)
    os.makedirs(plots_dir, exist_ok=True)

    model = load_model(checkpoint_path, device)

    all_results = []

    # ── CEDAR (writers 46-55) ─────────────────────────────────────────────
    cedar_gen, cedar_forg = build_cedar_test_pairs(cache_dir)
    all_results.append(
        _evaluate_one("CEDAR", cedar_gen, cedar_forg, model, device, plots_dir)
    )

    # ── Dataset2 (writers 651-686) ────────────────────────────────────────
    d2_gen, d2_forg = build_d2_test_pairs(cache_dir)
    if d2_gen and d2_forg:
        all_results.append(
            _evaluate_one("Dataset2", d2_gen, d2_forg, model, device, plots_dir)
        )
    else:
        print("\n  Dataset2 test cache not found — skipping D2 evaluation.")
        print("  (Run build_cache_dataset2 first, then re-evaluate.)")

    _print_summary(all_results)
    print(f"Plots saved to ./{PLOTS_DIR}/\n")

    return all_results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate signature model on CEDAR + Dataset2 test sets."
    )
    parser.add_argument(
        "--checkpoint", default=DEFAULT_CKPT,
        help=f"Path to .pt checkpoint. Default: {DEFAULT_CKPT}",
    )
    evaluate(checkpoint_path=parser.parse_args().checkpoint)
