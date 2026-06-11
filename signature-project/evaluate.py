"""
evaluate.py — Evaluation script for the trained Siamese network
================================================================
Loads the trained model, runs it on the 10 held-out test writers (46-55),
and measures how well genuine and forged signatures are separated.

What this script produces
-------------------------
1.  Console output
      • Number of pairs evaluated
      • EER (Equal Error Rate) and the cosine-similarity threshold at which it occurs

2.  plots/score_distributions.png
      Overlapping histograms: cosine-similarity scores for genuine-genuine pairs
      (should be high) vs genuine-forgery pairs (should be low).

3.  plots/far_frr_curve.png
      FAR and FRR plotted against every possible threshold, with the EER
      operating point marked.

Key concepts used here
----------------------
Cosine similarity  – dot product of two unit vectors; range [-1, 1].
                     1 = identical direction, -1 = opposite.
                     Because our embeddings are L2-normalised we use it directly.

FAR (False Acceptance Rate)  – fraction of forgery pairs whose similarity score
                                is ABOVE the decision threshold (model was fooled).

FRR (False Rejection Rate)   – fraction of genuine pairs whose similarity score
                                is BELOW the threshold (genuine signature rejected).

EER (Equal Error Rate)       – the threshold where FAR = FRR.
                                Lower EER = better model.  A random classifier
                                has EER ≈ 50 %.

Run
---
  python evaluate.py                         # uses checkpoints/best.pt by default
  python evaluate.py --checkpoint checkpoints/epoch_20.pt

Dependencies: torch, torchvision, numpy, matplotlib, opencv-python
"""

import os
import itertools
import argparse

import numpy as np
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")   # non-interactive backend — no display window needed
import matplotlib.pyplot as plt

from preprocess import preprocess_signature
from model      import SiameseNetwork
from dataset    import IMAGENET_MEAN, IMAGENET_STD


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Test writers are the 10 writers held out during training.
# They were never seen during training, so this gives an honest evaluation.
TEST_WRITERS   = list(range(46, 56))   # [46, 47, …, 55]
SAMPLES_EACH   = 24                    # 24 genuine + 24 forgery images per writer

EMBEDDING_DIM  = 128     # must match the value used in model.py / train.py
BATCH_SIZE     = 64      # images processed in one forward pass (larger = faster)

CHECKPOINT_DIR = "checkpoints"
PLOTS_DIR      = "plots"
DEFAULT_CKPT   = os.path.join(CHECKPOINT_DIR, "best.pt")


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(checkpoint_path: str, device: torch.device) -> SiameseNetwork:
    """
    Create the SiameseNetwork and load saved weights from a checkpoint file.

    train.py saves two kinds of files:
      best.pt / final.pt  — contain only the model weights dict
                            (saved with torch.save(model.state_dict(), path))
      epoch_NN.pt         — contain a larger dict with keys:
                            "model_state_dict", "optim_state_dict", "epoch", …

    This function handles both formats automatically.

    Parameters
    ----------
    checkpoint_path : str    path to the .pt file
    device          : torch.device

    Returns
    -------
    SiameseNetwork   ready for inference (in eval mode, on `device`)
    """
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            f"Run train.py first to generate it."
        )

    # torch.load reads the file; map_location ensures tensors land on the right
    # device regardless of where the model was originally trained.
    payload = torch.load(checkpoint_path, map_location=device)

    # Detect which format was saved.
    if isinstance(payload, dict) and "model_state_dict" in payload:
        # Full training checkpoint (epoch_NN.pt format)
        state_dict = payload["model_state_dict"]
        saved_epoch = payload.get("epoch", "?")
        saved_loss  = payload.get("loss",  float("nan"))
        print(f"Loaded epoch-{saved_epoch} checkpoint  (train loss: {saved_loss:.6f})")
    else:
        # Weights-only checkpoint (best.pt / final.pt format)
        state_dict  = payload
        print(f"Loaded weights-only checkpoint from {checkpoint_path}")

    # Build the model architecture, then fill in the saved weights.
    model = SiameseNetwork(embedding_dim=EMBEDDING_DIM)
    model.load_state_dict(state_dict)

    # .eval() switches BatchNorm and Dropout to inference mode.
    # Always call this before running a model for evaluation or inference.
    model.eval()
    model.to(device)
    return model


# ---------------------------------------------------------------------------
# Pair construction
# ---------------------------------------------------------------------------

def build_all_test_pairs(root_dir: str):
    """
    Build the *complete* set of evaluation pairs for test writers 46-55.

    Unlike the training dataset which sub-samples negatives for class balance,
    evaluation should use every possible pair so the score distributions are
    fully representative:

      Genuine-genuine pairs  — C(24, 2) = 276 per writer × 10 = 2 760 total
      Genuine-forgery pairs  — 24 × 24  = 576 per writer × 10 = 5 760 total

    Parameters
    ----------
    root_dir : str   folder containing full_org/ and full_forg/

    Returns
    -------
    genuine_pairs : list of (path_a, path_b)
    forgery_pairs : list of (path_a, path_b)
    """
    org_dir  = os.path.join(root_dir, "full_org")
    forg_dir = os.path.join(root_dir, "full_forg")

    genuine_pairs = []   # both images are genuine signatures of the same writer
    forgery_pairs = []   # one genuine + one skilled forgery of the same writer

    for writer_id in TEST_WRITERS:

        # Build the list of file paths for this writer.
        genuine_paths = [
            os.path.join(org_dir,  f"original_{writer_id}_{i}.png")
            for i in range(1, SAMPLES_EACH + 1)
        ]
        forgery_paths = [
            os.path.join(forg_dir, f"forgeries_{writer_id}_{i}.png")
            for i in range(1, SAMPLES_EACH + 1)
        ]

        # --- Genuine-genuine pairs -------------------------------------
        # itertools.combinations returns all unordered pairs without
        # repetition: (img1,img2), (img1,img3), … (img23,img24).
        # C(24, 2) = 276 pairs per writer.
        for path_a, path_b in itertools.combinations(genuine_paths, 2):
            genuine_pairs.append((path_a, path_b))

        # --- Genuine-forgery pairs -------------------------------------
        # Every genuine image paired with every forgery: 24 × 24 = 576.
        for path_g in genuine_paths:
            for path_f in forgery_paths:
                forgery_pairs.append((path_g, path_f))

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
    Run all pairs through the model and return cosine-similarity scores.

    Processing in batches is much faster than one pair at a time because the
    GPU (or CPU) can parallelise the forward pass across many images at once.

    Parameters
    ----------
    model      : trained SiameseNetwork in eval mode
    pairs      : list of (path_a, path_b)  tuples
    device     : torch.device
    batch_size : int  how many pairs to process together
    label      : str  name printed in progress messages ("genuine" / "forgery")

    Returns
    -------
    np.ndarray  shape [N]  cosine similarity in [-1, 1] for each pair
    """
    scores      = []
    n_pairs     = len(pairs)
    n_batches   = (n_pairs + batch_size - 1) // batch_size   # ceiling division

    # torch.no_grad() tells PyTorch not to build the computational graph for
    # backpropagation.  Since we are only doing inference (not training), this
    # saves memory and makes the forward pass faster.
    with torch.no_grad():

        for batch_idx in range(n_batches):
            # Slice out this batch's pairs.
            start = batch_idx * batch_size
            end   = min(start + batch_size, n_pairs)
            batch = pairs[start:end]

            # --- Load and preprocess every image in the batch -----------
            # preprocess_signature() returns a uint8 NumPy array (224×224).
            # We convert to float32, scale to [0,1], and add a channel dim.
            imgs_a, imgs_b = [], []
            for path_a, path_b in batch:
                arr_a = preprocess_signature(path_a).astype(np.float32) / 255.0
                arr_b = preprocess_signature(path_b).astype(np.float32) / 255.0
                # ImageNet normalisation — must match the normalisation applied
                # during training (dataset.py CEDARImageDataset.__getitem__).
                arr_a = (arr_a - IMAGENET_MEAN) / IMAGENET_STD
                arr_b = (arr_b - IMAGENET_MEAN) / IMAGENET_STD
                imgs_a.append(torch.from_numpy(arr_a).unsqueeze(0))
                imgs_b.append(torch.from_numpy(arr_b).unsqueeze(0))

            # torch.stack turns a list of [1,224,224] tensors into [B,1,224,224].
            imgs_a = torch.stack(imgs_a).to(device)
            imgs_b = torch.stack(imgs_b).to(device)

            # --- Forward pass -------------------------------------------
            emb_a, emb_b = model(imgs_a, imgs_b)   # each [B, 128], L2-normalised

            # Cosine similarity between each pair of embeddings: shape [B]
            # For L2-normalised vectors this equals the dot product.
            cos_sim = F.cosine_similarity(emb_a, emb_b, dim=1)

            # .cpu().numpy() moves the tensor back to CPU and converts to NumPy.
            scores.extend(cos_sim.cpu().numpy())

            # Progress indicator
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == n_batches:
                print(
                    f"  {label:8s}  batch {batch_idx+1:4d}/{n_batches}"
                    f"  ({end}/{n_pairs} pairs)",
                    end="\r",
                )

    print()   # newline after the \r progress line
    return np.array(scores, dtype=np.float32)


# ---------------------------------------------------------------------------
# EER computation
# ---------------------------------------------------------------------------

def compute_eer(
    genuine_scores: np.ndarray,
    forgery_scores: np.ndarray,
) -> tuple:
    """
    Compute the Equal Error Rate (EER) and the decision threshold at which it
    occurs.

    Approach
    --------
    We sweep through every unique cosine-similarity score as a possible
    decision threshold and compute FAR and FRR at each one.

      threshold τ: predict "genuine" if cosine_sim ≥ τ
                   predict "forgery" if cosine_sim < τ

      FAR(τ) = fraction of forgery pairs with score ≥ τ   (model was fooled)
      FRR(τ) = fraction of genuine pairs with score < τ   (genuine rejected)

    As τ rises:
      • FAR falls  (fewer forgeries pass the higher bar)
      • FRR rises  (more genuines fall below the higher bar)

    The EER is the threshold where FAR = FRR.  A lower EER means the model's
    score distributions overlap less, i.e., the model separates the two classes
    better.

    Implementation
    --------------
    Instead of looping over thresholds one by one (slow), we use numpy's
    searchsorted() which finds insertion positions in a sorted array in
    O(log N) time — effectively vectorising the loop.

    Parameters
    ----------
    genuine_scores : np.ndarray   cosine similarities for genuine-genuine pairs
    forgery_scores : np.ndarray   cosine similarities for genuine-forgery pairs

    Returns
    -------
    eer           : float   equal error rate in [0, 1]
    eer_threshold : float   the cosine-similarity threshold at the EER point
    thresholds    : np.ndarray   all threshold values used (for plotting)
    fars          : np.ndarray   FAR at each threshold
    frrs          : np.ndarray   FRR at each threshold
    """

    # Use every unique observed score as a candidate threshold.
    # This gives the finest possible resolution without any arbitrary binning.
    thresholds = np.sort(np.unique(
        np.concatenate([genuine_scores, forgery_scores])
    ))

    # Pre-sort both score arrays so searchsorted works correctly.
    forg_sorted = np.sort(forgery_scores)
    gen_sorted  = np.sort(genuine_scores)
    n_forg      = len(forgery_scores)
    n_gen       = len(genuine_scores)

    # np.searchsorted(sorted_array, values) returns the index at which
    # each value would need to be inserted to keep the array sorted.
    #
    # FAR(τ): how many forgery scores are ≥ τ?
    #   searchsorted gives how many scores are < τ (left side)
    #   so scores ≥ τ  =  total − left_count
    #
    # FRR(τ): how many genuine scores are < τ?
    #   that is exactly the left-side count from searchsorted
    far_counts = n_forg - np.searchsorted(forg_sorted, thresholds, side="left")
    frr_counts =          np.searchsorted(gen_sorted,  thresholds, side="left")

    fars = far_counts.astype(float) / n_forg
    frrs = frr_counts.astype(float) / n_gen

    # Find the threshold where |FAR − FRR| is smallest.
    abs_diff = np.abs(fars - frrs)
    idx      = int(np.argmin(abs_diff))

    # Refine: linearly interpolate between the two thresholds that straddle
    # the exact FAR = FRR crossing point (gives a more precise EER value).
    sign_changes = np.where(np.diff(np.sign(fars - frrs)))[0]
    if len(sign_changes) > 0:
        i = sign_changes[0]   # index just before the crossing

        # Solve FAR(t) = FRR(t) for t by parametric linear interpolation.
        # Let t = t_i + s*(t_{i+1} - t_i),  s in [0,1].
        # FAR(t) ≈ fars[i] + s*(fars[i+1] - fars[i])
        # FRR(t) ≈ frrs[i] + s*(frrs[i+1] - frrs[i])
        # Setting equal and solving for s:
        delta_far = fars[i + 1] - fars[i]
        delta_frr = frrs[i + 1] - frrs[i]
        denom     = delta_far - delta_frr

        s = (frrs[i] - fars[i]) / denom if abs(denom) > 1e-12 else 0.5

        eer_threshold = float(thresholds[i] + s * (thresholds[i + 1] - thresholds[i]))
        eer           = float(fars[i] + s * delta_far)
    else:
        # No exact crossing found — use the minimum-difference point.
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
    For each FAR target find the decision threshold and FRR.

    The operating point at FAR=X% is the LOWEST threshold where FAR<=X%.
    Lowest threshold = least restrictive gate that still stays within the
    FAR budget, which gives the lowest possible FRR at that FAR level.

    Parameters
    ----------
    thresholds  : sorted ascending cosine-similarity values
    fars        : FAR at each threshold (decreasing as threshold rises)
    frrs        : FRR at each threshold (increasing as threshold rises)
    far_targets : list of floats in [0, 1], e.g. [0.005, 0.01, 0.02, 0.05]

    Returns
    -------
    list of dicts with keys: far_target, threshold, actual_far, frr
    (all in [0, 1], not percentages)
    """
    rows = []
    for target in far_targets:
        valid = np.where(fars <= target)[0]
        if len(valid) == 0:
            rows.append(
                {"far_target": target, "threshold": None,
                 "actual_far": None,   "frr": None}
            )
        else:
            i = valid[0]   # lowest threshold achieving FAR <= target
            rows.append(
                {"far_target": target,
                 "threshold":  float(thresholds[i]),
                 "actual_far": float(fars[i]),
                 "frr":        float(frrs[i])}
            )
    return rows


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_score_distributions(
    genuine_scores: np.ndarray,
    forgery_scores: np.ndarray,
    eer_threshold:  float,
    plots_dir:      str,
    far1_threshold: float = None,
) -> None:
    """
    Draw overlapping histograms of cosine-similarity scores and save to PNG.

    Marks the EER threshold (black dashed) and the FAR=1% operating threshold
    (red dashed) so the bank's primary operating point is immediately visible.

    Parameters
    ----------
    genuine_scores  : scores for genuine-genuine pairs
    forgery_scores  : scores for genuine-forgery pairs
    eer_threshold   : cosine-similarity threshold at the EER point
    plots_dir       : output directory
    far1_threshold  : cosine-similarity threshold at FAR=1% (None = omit line)
    """
    fig, ax = plt.subplots(figsize=(9, 5))

    bins = np.linspace(-1.0, 1.0, 61)

    ax.hist(
        genuine_scores, bins=bins,
        color="steelblue", alpha=0.55,
        label=f"Genuine-Genuine  (n={len(genuine_scores):,})",
    )
    ax.hist(
        forgery_scores, bins=bins,
        color="tomato", alpha=0.55,
        label=f"Genuine-Forgery  (n={len(forgery_scores):,})",
    )

    ax.axvline(
        eer_threshold, color="black", linestyle="--", linewidth=1.5,
        label=f"EER threshold = {eer_threshold:.3f}",
    )

    if far1_threshold is not None:
        ax.axvline(
            far1_threshold, color="red", linestyle="--", linewidth=1.5,
            label=f"FAR=1% threshold = {far1_threshold:.3f}",
        )

    ax.set_xlabel("Cosine Similarity", fontsize=12)
    ax.set_ylabel("Number of Pairs",   fontsize=12)
    ax.set_title(
        "Score Distributions — Genuine vs Forgery Pairs\n"
        "(Test Writers 46-55, CEDAR dataset)",
        fontsize=12,
    )
    ax.legend(fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.4)

    out_path = os.path.join(plots_dir, "score_distributions.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out_path}")


def plot_far_frr_curve(
    thresholds:    np.ndarray,
    fars:          np.ndarray,
    frrs:          np.ndarray,
    eer:           float,
    eer_threshold: float,
    plots_dir:     str,
) -> None:
    """
    Plot FAR and FRR against the decision threshold and mark the EER crossing.

    The two curves cross exactly once.  The x-coordinate of that crossing is
    the EER threshold; the y-coordinate is the EER value itself.

    Parameters
    ----------
    thresholds    : threshold values used (x axis)
    fars, frrs    : error rates at each threshold
    eer           : equal error rate value (y coordinate of crossing)
    eer_threshold : cosine-similarity threshold at the EER (x coordinate)
    plots_dir     : output directory
    """
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(thresholds, fars,  color="tomato",    linewidth=2, label="FAR (False Acceptance Rate)")
    ax.plot(thresholds, frrs,  color="steelblue", linewidth=2, label="FRR (False Rejection Rate)")

    # Mark the EER operating point with a filled circle.
    ax.scatter(
        [eer_threshold], [eer],
        color="black", zorder=5, s=80,
        label=f"EER = {eer * 100:.2f} %  @ threshold {eer_threshold:.3f}",
    )

    # Dashed lines from the EER point to both axes make it easy to read off.
    ax.axvline(eer_threshold, color="black", linestyle=":", linewidth=1)
    ax.axhline(eer,           color="black", linestyle=":", linewidth=1)

    ax.set_xlabel("Decision Threshold (cosine similarity)", fontsize=12)
    ax.set_ylabel("Error Rate", fontsize=12)
    ax.set_title(
        "FAR / FRR vs Threshold — Equal Error Rate\n"
        "(Test Writers 46-55, CEDAR dataset)",
        fontsize=12,
    )
    ax.legend(fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-0.02, 1.02)

    out_path = os.path.join(plots_dir, "far_frr_curve.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out_path}")


# ---------------------------------------------------------------------------
# Main evaluation routine
# ---------------------------------------------------------------------------

def evaluate(checkpoint_path: str = DEFAULT_CKPT) -> None:
    """
    Full evaluation pipeline: load model → score all test pairs → EER → plots.
    """

    # ----------------------------------------------------------------
    # Device selection
    # ----------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device          : {device}")
    print(f"Checkpoint      : {checkpoint_path}")
    print("-" * 50)

    # ----------------------------------------------------------------
    # Create output folder for plots
    # ----------------------------------------------------------------
    script_dir = os.path.dirname(os.path.abspath(__file__))
    plots_dir  = os.path.join(script_dir, PLOTS_DIR)
    os.makedirs(plots_dir, exist_ok=True)

    # ----------------------------------------------------------------
    # Load the trained model
    # ----------------------------------------------------------------
    model = load_model(checkpoint_path, device)

    # ----------------------------------------------------------------
    # Build all evaluation pairs for test writers
    # ----------------------------------------------------------------
    print("Building test pairs …")
    genuine_pairs, forgery_pairs = build_all_test_pairs(script_dir)
    print(f"  Genuine-genuine pairs : {len(genuine_pairs):,}")
    print(f"  Genuine-forgery pairs : {len(forgery_pairs):,}")
    print("-" * 50)

    # ----------------------------------------------------------------
    # Run inference — collect a cosine-similarity score for every pair
    # ----------------------------------------------------------------
    print("Scoring genuine-genuine pairs …")
    genuine_scores = compute_scores(model, genuine_pairs, device, BATCH_SIZE, "genuine")

    print("Scoring genuine-forgery pairs …")
    forgery_scores = compute_scores(model, forgery_pairs, device, BATCH_SIZE, "forgery")

    # ----------------------------------------------------------------
    # Print basic statistics to get a feel for the distributions
    # ----------------------------------------------------------------
    print("-" * 50)
    print(f"Genuine-genuine  mean={genuine_scores.mean():.4f}  "
          f"std={genuine_scores.std():.4f}  "
          f"min={genuine_scores.min():.4f}  max={genuine_scores.max():.4f}")
    print(f"Genuine-forgery  mean={forgery_scores.mean():.4f}  "
          f"std={forgery_scores.std():.4f}  "
          f"min={forgery_scores.min():.4f}  max={forgery_scores.max():.4f}")
    print("-" * 50)

    # ----------------------------------------------------------------
    # Compute EER
    # ----------------------------------------------------------------
    print("Computing EER and operating points …")
    eer, eer_threshold, thresholds, fars, frrs = compute_eer(
        genuine_scores, forgery_scores
    )

    print(f"\n  EER            : {eer * 100:.2f} %")
    print(f"  EER threshold  : {eer_threshold:.4f}  (cosine similarity)")
    print(
        f"\n  Interpretation : at this threshold the model incorrectly "
        f"rejects {eer*100:.1f} % of genuine pairs\n"
        f"                   and incorrectly accepts {eer*100:.1f} % of "
        f"forgery pairs."
    )
    print("-" * 50)

    # ----------------------------------------------------------------
    # Operating-point table (bank's primary metric: FAR against forgeries)
    # ----------------------------------------------------------------
    FAR_TARGETS = [0.005, 0.01, 0.02, 0.05]
    op_points   = compute_operating_points(thresholds, fars, frrs, FAR_TARGETS)

    print("\n  Operating points (FAR measured against skilled forgeries):\n")
    print(f"  {'FAR target':>10}  {'Threshold':>10}  {'Actual FAR':>11}  {'FRR (miss)':>11}")
    print(f"  {'-'*10}  {'-'*10}  {'-'*11}  {'-'*11}")
    for row in op_points:
        if row["threshold"] is None:
            print(f"  {row['far_target']*100:>9.1f}%  {'N/A':>10}  {'N/A':>11}  {'N/A':>11}")
        else:
            print(
                f"  {row['far_target']*100:>9.1f}%  "
                f"{row['threshold']:>10.4f}  "
                f"{row['actual_far']*100:>10.2f}%  "
                f"{row['frr']*100:>10.2f}%"
            )
    print()
    print("-" * 50)

    # Extract the FAR=1% threshold for the histogram marker
    far1_row       = next((r for r in op_points if r["far_target"] == 0.01), None)
    far1_threshold = far1_row["threshold"] if far1_row else None

    # ----------------------------------------------------------------
    # Save plots
    # ----------------------------------------------------------------
    print("Saving plots …")
    plot_score_distributions(
        genuine_scores, forgery_scores,
        eer_threshold, plots_dir,
        far1_threshold=far1_threshold,
    )
    plot_far_frr_curve(thresholds, fars, frrs, eer, eer_threshold, plots_dir)

    print("-" * 50)
    print(f"Done.  Plots saved to ./{PLOTS_DIR}/")


# ---------------------------------------------------------------------------
# Entry point — supports an optional --checkpoint argument
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate trained Siamese network on CEDAR test writers."
    )
    parser.add_argument(
        "--checkpoint",
        default=DEFAULT_CKPT,
        help=(
            f"Path to a .pt checkpoint file.  "
            f"Accepts both weights-only files (best.pt, final.pt) and "
            f"full training checkpoints (epoch_NN.pt).  "
            f"Default: {DEFAULT_CKPT}"
        ),
    )
    args = parser.parse_args()
    evaluate(checkpoint_path=args.checkpoint)
