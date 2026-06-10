"""
train.py — Training script for the Siamese signature-verification network
=========================================================================
Trains the SiameseNetwork (model.py) on the CEDAR training writers using
Contrastive Loss measured on cosine distance.

High-level flow
---------------
  Dataset  → pairs (img1, img2, label)          [CEDARSignatureDataset]
  Model    → (emb_a, emb_b)  L2-normalised      [SiameseNetwork]
  Loss     → contrastive loss on cosine distance  [contrastive_loss()]
  Optimiser→ Adam  lr=1e-4

Output (all written to  checkpoints/ )
---------------------------------------
  epoch_01.pt … epoch_20.pt  — full checkpoint after every epoch
  best.pt                    — weights-only snapshot of the best epoch
  final.pt                   — weights-only snapshot after the last epoch
  loss_curve.png             — training loss plotted against epoch number

Run
---
  python train.py

Dependencies: torch, torchvision, matplotlib, opencv-python, numpy
"""

import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Use the non-interactive "Agg" backend so matplotlib never tries to open
# a display window.  This is especially important on remote servers / CI.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Local modules in this project
from dataset import CEDARSignatureDataset
from model   import SiameseNetwork


# ---------------------------------------------------------------------------
# Hyperparameters
# All tunable values are collected here so you never have to hunt through
# the code to change them.
# ---------------------------------------------------------------------------

BATCH_SIZE     = 32       # how many pairs the model sees at once
LEARNING_RATE  = 1e-4     # Adam step size; 0.0001 is a safe default for fine-tuning
NUM_EPOCHS     = 20       # number of full passes through the training set
MARGIN         = 0.5      # contrastive loss margin (cosine-distance units, range [0, 2])
EMBEDDING_DIM  = 128      # must match the value in model.py
SEED           = 42       # fixes randomness for reproducibility

CHECKPOINT_DIR = "checkpoints"   # all outputs go here


# ---------------------------------------------------------------------------
# Contrastive Loss  (Hadsell et al., 2006)  on cosine distance
# ---------------------------------------------------------------------------

def contrastive_loss(
    emb_a: torch.Tensor,
    emb_b: torch.Tensor,
    labels: torch.Tensor,
    margin: float = 0.5,
) -> torch.Tensor:
    """
    Compute the contrastive loss for a batch of embedding pairs.

    Why cosine distance?
    --------------------
    Our model L2-normalises every embedding, so all vectors have length 1 and
    live on the surface of a unit hypersphere.  For unit vectors the cosine
    similarity equals the dot product, and cosine distance = 1 − cosine_sim.

    Cosine distance range:
      0   → embeddings point in exactly the same direction  (very similar)
      1   → embeddings are orthogonal                       (unrelated)
      2   → embeddings point in opposite directions         (maximally different)

    Loss formula (one pair)
    -----------------------
    Let  d  = cosine distance between emb_a and emb_b.

      label = 1  (genuine pair — should be close):
        loss_genuine = d²
        The model is penalised the further apart a genuine pair is.

      label = 0  (forgery pair — should be far apart):
        loss_forgery = max(0, margin − d)²
        The model is penalised only when a forgery pair is CLOSER than `margin`.
        Once the pair is already beyond `margin`, the loss is zero — no need
        to push further.

    Total loss = mean over all pairs in the batch.

    Parameters
    ----------
    emb_a, emb_b : torch.Tensor   shape [B, D], L2-normalised embeddings
    labels       : torch.Tensor   shape [B],    1 = genuine, 0 = forgery
    margin       : float          minimum desired distance for forgery pairs

    Returns
    -------
    torch.Tensor   scalar — mean loss over the batch
    """

    # F.cosine_similarity computes the dot product of unit vectors: shape [B]
    # Values range from -1.0 (opposite) through 0.0 (orthogonal) to 1.0 (identical).
    cos_sim  = F.cosine_similarity(emb_a, emb_b, dim=1)

    # Flip to distance:  0 = identical … 2 = maximally different
    distance = 1.0 - cos_sim   # shape [B]

    # Cast labels to float so they can be used as multipliers in arithmetic.
    # y = 1.0 for genuine pairs, 0.0 for forgery pairs.
    y = labels.float()

    # Genuine-pair term: active when y = 1.
    # Squared distance penalises pairs that are too far apart.
    loss_genuine = y * distance.pow(2)

    # Forgery-pair term: active when y = 0.
    # F.relu(x) = max(0, x) — zeroes out any negative values.
    # If the distance is already larger than margin, the contribution is 0.
    loss_forgery = (1.0 - y) * F.relu(margin - distance).pow(2)

    # Average across the batch to produce one scalar we can call .backward() on.
    return (loss_genuine + loss_forgery).mean()


# ---------------------------------------------------------------------------
# Loss-curve helper
# ---------------------------------------------------------------------------

def _save_loss_curve(epoch_losses: list, out_dir: str) -> None:
    """
    Plot training loss vs. epoch and save the figure as a PNG.

    Parameters
    ----------
    epoch_losses : list of float   one value per epoch
    out_dir      : str             folder where loss_curve.png is written
    """

    epochs = range(1, len(epoch_losses) + 1)

    fig, ax = plt.subplots(figsize=(8, 5))

    # Main line + circle markers at each epoch
    ax.plot(
        epochs, epoch_losses,
        marker="o", linewidth=2, markersize=5,
        color="steelblue", label="Training loss",
    )

    # Highlight the best (lowest-loss) epoch with a red dot
    best_idx   = epoch_losses.index(min(epoch_losses))  # 0-based index
    best_epoch = best_idx + 1                           # 1-based epoch number
    ax.scatter(
        best_epoch, epoch_losses[best_idx],
        color="red", zorder=5, s=80,
        label=f"Best: epoch {best_epoch}  (loss {epoch_losses[best_idx]:.4f})",
    )

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Contrastive Loss", fontsize=12)
    ax.set_title("Training Loss — Siamese Signature Verification", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5)

    plot_path = os.path.join(out_dir, "loss_curve.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)   # release memory; important in long training runs

    print(f"Loss curve saved  →  {plot_path}")


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train() -> None:
    """
    Build dataset / model / optimiser, run the training loop, save outputs.
    """

    # ----------------------------------------------------------------
    # Reproducibility
    # Fixing the seed makes every run produce identical results, which
    # is important when comparing experiments.
    # ----------------------------------------------------------------
    torch.manual_seed(SEED)

    # ----------------------------------------------------------------
    # Device selection
    # A CUDA-capable GPU can be 10-50× faster than a CPU.
    # torch.cuda.is_available() returns False if no GPU is present, so
    # the code automatically falls back to CPU.
    # ----------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device         : {device}")

    # ----------------------------------------------------------------
    # Create the checkpoint output directory (no error if it exists).
    # ----------------------------------------------------------------
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # ----------------------------------------------------------------
    # Dataset
    #
    # CEDARSignatureDataset builds all (img1, img2, label) pairs at
    # construction time.  We use split="train" which restricts the data
    # to writers 1-45 (writers 46-55 are held out for testing).
    #
    # root_dir is the folder that contains full_org/ and full_forg/.
    # Using __file__ makes the path relative to this script, not the
    # working directory from which you run it.
    # ----------------------------------------------------------------
    script_dir    = os.path.dirname(os.path.abspath(__file__))
    train_dataset = CEDARSignatureDataset(
        root_dir=script_dir,
        split="train",
        seed=SEED,
    )

    # ----------------------------------------------------------------
    # DataLoader
    #
    # Wraps the dataset to:
    #   • Group samples into batches of BATCH_SIZE
    #   • Shuffle pairs at the start of each epoch (shuffle=True)
    #   • Optionally load data in parallel (num_workers)
    #
    # num_workers=0: load data in the main process.  Safe on Windows
    # because multiprocessing in DataLoader requires the  __main__
    # guard, which we have, but 0 avoids any platform-specific issues.
    # Increase to 2 or 4 on Linux/macOS for faster data loading.
    #
    # pin_memory=True: when using a GPU, keeps the data in "pinned"
    # (page-locked) CPU memory so the GPU can copy it faster.
    # ----------------------------------------------------------------
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    n_pairs   = len(train_dataset)
    n_batches = len(train_loader)
    print(f"Training pairs : {n_pairs}")
    print(f"Batches/epoch  : {n_batches}")
    print(f"Epochs         : {NUM_EPOCHS}")
    print(f"Batch size     : {BATCH_SIZE}")
    print(f"Learning rate  : {LEARNING_RATE}")
    print(f"Margin         : {MARGIN}")
    print("-" * 45)

    # ----------------------------------------------------------------
    # Model
    #
    # .to(device) moves all model parameters to the chosen device
    # (GPU memory or CPU RAM).  Every input tensor must be on the
    # same device as the model — we handle that in the training loop.
    # ----------------------------------------------------------------
    model = SiameseNetwork(embedding_dim=EMBEDDING_DIM).to(device)

    # ----------------------------------------------------------------
    # Optimiser
    #
    # Adam (Adaptive Moment Estimation) maintains a per-parameter
    # learning rate and uses momentum, making it robust to noisy
    # gradients and fast to converge.  lr=1e-4 is a common default
    # for fine-tuning a pretrained backbone.
    # ----------------------------------------------------------------
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # ----------------------------------------------------------------
    # Training loop
    # ----------------------------------------------------------------
    epoch_losses = []          # collected for the loss-curve plot
    best_loss    = float("inf")

    for epoch in range(1, NUM_EPOCHS + 1):

        # model.train() activates layers that behave differently
        # during training vs evaluation, such as Dropout and BatchNorm.
        # (ResNet-18 uses BatchNorm, so this matters.)
        model.train()

        running_loss = 0.0   # sum of batch losses within this epoch
        n_seen       = 0     # number of batches processed

        for img1, img2, labels in train_loader:

            # Move tensors to the same device as the model.
            # This is a no-op if already on the right device.
            img1   = img1.to(device)    # [B, 1, 224, 224]
            img2   = img2.to(device)    # [B, 1, 224, 224]
            labels = labels.to(device)  # [B]

            # ------ Forward pass ------------------------------------
            # The Siamese network runs both images through the shared
            # backbone and returns two L2-normalised embeddings.
            emb_a, emb_b = model(img1, img2)   # each [B, 128]

            # Compute the contrastive loss for this batch.
            loss = contrastive_loss(emb_a, emb_b, labels, margin=MARGIN)

            # ------ Backward pass + optimiser step ------------------
            # Step 1: clear gradients from the previous iteration.
            #         PyTorch accumulates gradients by default; we must
            #         reset them each step or they would pile up.
            optimizer.zero_grad()

            # Step 2: backpropagate — compute how much each weight
            #         contributed to the loss (∂loss/∂weight).
            loss.backward()

            # Step 3: update every weight by a small step in the
            #         direction that reduces the loss.
            optimizer.step()

            running_loss += loss.item()   # .item() extracts a plain Python float
            n_seen       += 1

        # ---- Epoch summary -----------------------------------------
        epoch_loss = running_loss / n_seen   # mean loss over all batches
        epoch_losses.append(epoch_loss)

        print(f"Epoch [{epoch:2d}/{NUM_EPOCHS}]  Loss: {epoch_loss:.6f}")

        # ---- Save per-epoch checkpoint -----------------------------
        # We save both the model state and the optimiser state.  The
        # optimiser state contains momentum buffers, so saving it lets
        # you resume training from this point with no warm-up cost.
        ckpt_path = os.path.join(CHECKPOINT_DIR, f"epoch_{epoch:02d}.pt")
        torch.save(
            {
                "epoch"           : epoch,
                "model_state_dict": model.state_dict(),
                "optim_state_dict": optimizer.state_dict(),
                "loss"            : epoch_loss,
                "margin"          : MARGIN,
                "embedding_dim"   : EMBEDDING_DIM,
            },
            ckpt_path,
        )

        # ---- Track best model --------------------------------------
        # "Best" = lowest training loss so far.  We save only the
        # model weights (not the optimiser) because best.pt is
        # intended for inference, not for resuming training.
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(
                model.state_dict(),
                os.path.join(CHECKPOINT_DIR, "best.pt"),
            )
            print(f"           ↳  New best saved  (loss: {best_loss:.6f})")

    # ----------------------------------------------------------------
    # Post-training outputs
    # ----------------------------------------------------------------

    # Save final model weights (convenient when you just want to load
    # the last checkpoint without knowing which epoch number it was).
    torch.save(
        model.state_dict(),
        os.path.join(CHECKPOINT_DIR, "final.pt"),
    )

    print("-" * 45)
    print(f"Training complete.")
    print(f"Best loss     : {best_loss:.6f}")
    print(f"Checkpoints   : ./{CHECKPOINT_DIR}/")

    # Save the loss curve plot.
    _save_loss_curve(epoch_losses, CHECKPOINT_DIR)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
# The  if __name__ == "__main__":  guard ensures that train() is only called
# when you run this file directly (python train.py), NOT when another module
# imports it.  This is also required for DataLoader's multiprocessing on
# Windows.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    train()
