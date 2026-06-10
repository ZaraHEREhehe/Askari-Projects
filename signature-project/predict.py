"""
predict.py — Single-pair signature verification
================================================
Given two signature images, this script tells you whether they are likely
written by the same person (GENUINE MATCH) or not (FORGERY / MISMATCH),
and prints a human-readable similarity percentage.

Usage
-----
  # Basic — compare two images using the best trained model
  python predict.py path/to/sig_A.png path/to/sig_B.png

  # Override the decision threshold (default 0.5)
  python predict.py sig_A.png sig_B.png --threshold 0.6

  # Use a specific checkpoint
  python predict.py sig_A.png sig_B.png --checkpoint checkpoints/epoch_15.pt

  # Also save a side-by-side comparison image
  python predict.py sig_A.png sig_B.png --save-plot

How it works (plain English)
-----------------------------
1. Both images are cleaned and resized to 224×224.
2. The trained network converts each image into a 128-number "fingerprint"
   (embedding vector).
3. Cosine similarity is computed between the two fingerprints.
   • Values close to +1  → fingerprints point in the same direction → similar
   • Values close to  0  → fingerprints are unrelated → different
4. If similarity ≥ threshold  →  GENUINE MATCH
   If similarity <  threshold  →  FORGERY / MISMATCH

Choosing a threshold
--------------------
The default threshold (0.5) matches the training margin.
For the *optimal* threshold, run evaluate.py first — it reports the EER
threshold, which minimises the combined error rate on the test set.

Dependencies: torch, torchvision, numpy, matplotlib, opencv-python
"""

import os
import argparse

import cv2
import numpy as np
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")   # write PNG files without needing a display
import matplotlib.pyplot as plt

from preprocess import preprocess_signature
from model      import SiameseNetwork


# ---------------------------------------------------------------------------
# Configuration defaults
# ---------------------------------------------------------------------------

EMBEDDING_DIM     = 128    # must match model.py / train.py
DEFAULT_CKPT      = os.path.join("checkpoints", "best.pt")
DEFAULT_THRESHOLD = 0.5    # cosine similarity cut-off for genuine vs forgery
                           # tip: replace with the EER threshold from evaluate.py


# ---------------------------------------------------------------------------
# Model loading  (same pattern as evaluate.py)
# ---------------------------------------------------------------------------

def load_model(checkpoint_path: str, device: torch.device) -> SiameseNetwork:
    """
    Build the SiameseNetwork and load saved weights from a checkpoint.

    Handles both file formats produced by train.py:
      • best.pt / final.pt  — weights-only dict
      • epoch_NN.pt         — full training dict with key "model_state_dict"

    Parameters
    ----------
    checkpoint_path : str
    device          : torch.device

    Returns
    -------
    SiameseNetwork in eval mode, on `device`
    """
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            "Train the model first with:  python train.py"
        )

    # Load whatever was saved — could be a plain state_dict or a larger dict.
    payload = torch.load(checkpoint_path, map_location=device)

    if isinstance(payload, dict) and "model_state_dict" in payload:
        state_dict = payload["model_state_dict"]
    else:
        state_dict = payload

    model = SiameseNetwork(embedding_dim=EMBEDDING_DIM)
    model.load_state_dict(state_dict)
    model.eval()          # switch BatchNorm / Dropout to inference mode
    model.to(device)
    return model


# ---------------------------------------------------------------------------
# Core prediction function
# ---------------------------------------------------------------------------

def predict(
    path_a:     str,
    path_b:     str,
    model:      SiameseNetwork,
    device:     torch.device,
    threshold:  float = DEFAULT_THRESHOLD,
) -> dict:
    """
    Compare two signature images and return the verification result.

    Parameters
    ----------
    path_a, path_b : str     paths to the two signature image files
    model          : trained SiameseNetwork
    device         : torch.device
    threshold      : float   cosine-similarity cut-off  (genuine if ≥ threshold)

    Returns
    -------
    dict with keys:
      "cosine_similarity" : float   raw score in [-1, 1]
      "similarity_pct"    : float   rescaled to [0, 100]
      "decision"          : str     "GENUINE MATCH" or "FORGERY / MISMATCH"
      "is_genuine"        : bool
      "threshold"         : float   the threshold that was used
    """

    # ----------------------------------------------------------------
    # Step 1 — Preprocess both images into clean 224×224 tensors
    # ----------------------------------------------------------------
    # preprocess_signature returns a uint8 numpy array (224, 224).
    # We cast to float32, scale 0-255 → 0.0-1.0, and add a channel dim
    # so the shape becomes (1, 224, 224) as the model expects.
    def _load(path: str) -> torch.Tensor:
        arr = preprocess_signature(path).astype(np.float32) / 255.0
        # unsqueeze(0) adds a batch dimension: (1,224,224) → (1,1,224,224)
        return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)

    img_a = _load(path_a).to(device)   # shape [1, 1, 224, 224]
    img_b = _load(path_b).to(device)

    # ----------------------------------------------------------------
    # Step 2 — Get embeddings from the model
    # ----------------------------------------------------------------
    # torch.no_grad() tells PyTorch we are not training, so it skips
    # storing information needed for backpropagation.  This saves memory
    # and speeds up the forward pass.
    with torch.no_grad():
        emb_a, emb_b = model(img_a, img_b)   # each shape [1, 128]

    # ----------------------------------------------------------------
    # Step 3 — Compute cosine similarity
    # ----------------------------------------------------------------
    # Both embeddings are already L2-normalised by the model, so cosine
    # similarity is simply their dot product.  Result is a single number
    # in the range [-1.0, 1.0].
    cos_sim = F.cosine_similarity(emb_a, emb_b, dim=1).item()
    # .item() converts a 1-element tensor to a plain Python float.

    # ----------------------------------------------------------------
    # Step 4 — Rescale to a human-friendly 0-100 % range
    # ----------------------------------------------------------------
    # cos_sim lives in [-1, 1].
    # Adding 1 shifts it to [0, 2], dividing by 2 gives [0, 1],
    # multiplying by 100 gives [0 %, 100 %].
    similarity_pct = (cos_sim + 1.0) / 2.0 * 100.0

    # ----------------------------------------------------------------
    # Step 5 — Apply the threshold to make a binary decision
    # ----------------------------------------------------------------
    is_genuine = cos_sim >= threshold
    decision   = "GENUINE MATCH" if is_genuine else "FORGERY / MISMATCH"

    return {
        "cosine_similarity": cos_sim,
        "similarity_pct":    similarity_pct,
        "decision":          decision,
        "is_genuine":        is_genuine,
        "threshold":         threshold,
    }


# ---------------------------------------------------------------------------
# Optional side-by-side comparison plot
# ---------------------------------------------------------------------------

def save_comparison_plot(
    path_a:  str,
    path_b:  str,
    result:  dict,
    out_dir: str = ".",
) -> str:
    """
    Save a side-by-side figure of both preprocessed signatures with the
    verification verdict printed as the title.

    Parameters
    ----------
    path_a, path_b : str    input image paths
    result         : dict   output of predict()
    out_dir        : str    folder where the PNG is saved

    Returns
    -------
    str   path to the saved PNG file
    """
    # Preprocess both images for display (same pipeline as model input).
    arr_a = preprocess_signature(path_a)   # uint8, shape (224, 224)
    arr_b = preprocess_signature(path_b)

    # Choose a border colour based on the verdict:
    #   green = genuine match,  red = forgery / mismatch
    border_color = "green" if result["is_genuine"] else "red"

    fig, axes = plt.subplots(1, 2, figsize=(8, 4.5))

    for ax, arr, path, label in zip(
        axes,
        [arr_a, arr_b],
        [path_a, path_b],
        ["Signature A", "Signature B"],
    ):
        ax.imshow(arr, cmap="gray", vmin=0, vmax=255)
        ax.set_title(label + f"\n({os.path.basename(path)})", fontsize=10)
        ax.axis("off")

        # Draw a coloured border around each image panel using the axes spines.
        for spine in ax.spines.values():
            spine.set_edgecolor(border_color)
            spine.set_linewidth(4)
            spine.set_visible(True)

    # Main title carries the verdict and score.
    verdict_line = (
        f"{result['decision']}\n"
        f"Similarity: {result['similarity_pct']:.1f} %  "
        f"(cosine: {result['cosine_similarity']:.4f},  "
        f"threshold: {result['threshold']:.3f})"
    )
    fig.suptitle(verdict_line, fontsize=12, color=border_color, fontweight="bold")

    fig.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "prediction.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Console output helper
# ---------------------------------------------------------------------------

def print_result(result: dict, path_a: str, path_b: str) -> None:
    """Print a clean, readable summary of the prediction result."""
    width = 50
    is_genuine = result["is_genuine"]

    print()
    print("=" * width)
    print("  SIGNATURE VERIFICATION RESULT")
    print("=" * width)
    print(f"  Image A       :  {os.path.basename(path_a)}")
    print(f"  Image B       :  {os.path.basename(path_b)}")
    print("-" * width)
    print(f"  Cosine sim    :  {result['cosine_similarity']:+.4f}  (range -1 to +1)")
    print(f"  Similarity    :  {result['similarity_pct']:.1f} %")
    print(f"  Threshold     :  {result['threshold']:.3f}")
    print("-" * width)

    # Make the verdict stand out visually.
    verdict = result["decision"]
    padding = (width - len(verdict) - 4) // 2
    print(f"  {'':>{padding}}{verdict}")

    print("=" * width)
    print()


# ---------------------------------------------------------------------------
# Entry point — command-line interface
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Verify whether two signatures belong to the same writer.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python predict.py full_org/original_46_1.png full_org/original_46_2.png\n"
            "  python predict.py sig1.png sig2.png --threshold 0.62 --save-plot\n"
            "  python predict.py sig1.png sig2.png --checkpoint checkpoints/epoch_20.pt"
        ),
    )

    parser.add_argument("image_a", help="Path to the first signature image.")
    parser.add_argument("image_b", help="Path to the second signature image.")
    parser.add_argument(
        "--checkpoint",
        default=DEFAULT_CKPT,
        help=f"Path to model checkpoint.  Default: {DEFAULT_CKPT}",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=(
            f"Cosine-similarity cut-off.  Pairs scoring >= threshold are "
            f"accepted as genuine.  Default: {DEFAULT_THRESHOLD}  "
            f"(tip: use the EER threshold printed by evaluate.py for best accuracy)"
        ),
    )
    parser.add_argument(
        "--save-plot",
        action="store_true",
        help="Save a side-by-side comparison image to prediction.png.",
    )

    args = parser.parse_args()

    # --- Validate input paths -------------------------------------------
    for p in (args.image_a, args.image_b):
        if not os.path.isfile(p):
            raise FileNotFoundError(f"Image not found: {p}")

    # --- Device ---------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Load model -----------------------------------------------------
    model = load_model(args.checkpoint, device)

    # --- Run prediction -------------------------------------------------
    result = predict(args.image_a, args.image_b, model, device, args.threshold)

    # --- Print result ---------------------------------------------------
    print_result(result, args.image_a, args.image_b)

    # --- Optionally save plot -------------------------------------------
    if args.save_plot:
        out_path = save_comparison_plot(args.image_a, args.image_b, result)
        print(f"Comparison image saved  →  {out_path}\n")
