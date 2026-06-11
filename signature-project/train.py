"""
train.py — Triplet-loss training with online semi-hard negative mining
=======================================================================
Trains SiameseNetwork (model.py) on CEDAR writers 1-45.

Key changes from the contrastive-loss version
----------------------------------------------
Loss    : triplet loss, margin=0.3 on cosine distance, online semi-hard mining.
          The old contrastive margin=0.5 dead-zone let training writers saturate
          in one epoch; triplet semi-hard mining keeps all epochs productive.
Data    : disk-cached preprocessed images (build_cache runs once).
          PKSampler re-draws fresh writer/image combinations every epoch so
          there is no fixed pair list for the model to memorise.
Norm    : ImageNet mean/std applied to all inputs — required for pretrained
          ResNet-18 features to operate in their expected activation range.
LR      : Adam 1e-4, halved after epoch LR_DECAY_EPOCH via MultiStepLR.

Run
---
  python train.py            # full 10-epoch training run
  python train.py --sanity   # 1 epoch, 4 batches — smoke test only
"""

import os
import time
import argparse
from datetime import datetime

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataset import build_cache, CEDARImageDataset, PKSampler, TRAIN_WRITERS
from model   import SiameseNetwork


# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

LEARNING_RATE         = 1e-4
NUM_EPOCHS            = 10
MARGIN                = 0.3   # cosine-distance triplet margin; tighter than the old 0.5
EMBEDDING_DIM         = 128
SEED                  = 42

P                     = 16    # writers sampled per batch
K_GENUINE             = 4     # genuine images per writer per batch
K_FORGERY             = 4     # forgery images per writer per batch
# ~94 batches x 16 writers x 4 genuine anchors ≈ 6 000 triplets/epoch
NUM_BATCHES_PER_EPOCH = 94

LR_DECAY_EPOCH        = 5     # LR halved after this epoch (effect from epoch 6)
LR_DECAY_FACTOR       = 0.5

CHECKPOINT_DIR        = "checkpoints"
CACHE_DIR             = "cache"
NUM_WORKERS           = 4     # set to 0 if multiprocessing causes issues on Windows


# ---------------------------------------------------------------------------
# Triplet loss — online semi-hard negative mining
# ---------------------------------------------------------------------------

def triplet_loss_semihard(
    embeddings: torch.Tensor,
    writer_ids,
    is_genuine,
    margin: float = 0.3,
) -> tuple:
    """
    Vectorised online semi-hard negative mining.

    anchor   = genuine image i
    positive = same-writer genuine j (hardest: max cosine distance from anchor)
    negative = same-writer skilled forgery k  (primary — the actual threat)
               OR different-writer genuine k  (secondary — easy regulariser)
               Selection: semi-hard (pos_dist < neg_dist < pos_dist + margin).
               Fallback: closest valid negative when no semi-hard exists.

    Using same-writer forgeries as the primary negative pool forces the model to
    learn features that separate a writer's own genuine from their own forgeries,
    not just features that separate different writers from each other.

    Returns (loss_scalar, n_valid_triplets)
    """
    device  = embeddings.device
    N       = len(embeddings)

    wids    = torch.as_tensor(writer_ids, dtype=torch.long, device=device)
    is_gen  = torch.as_tensor(is_genuine, dtype=torch.bool, device=device)
    is_forg = ~is_gen

    # Pairwise cosine distance; embeddings are L2-normalised so dot = cosine sim
    sim  = torch.mm(embeddings, embeddings.t()).clamp(-1.0, 1.0)
    dist = 1.0 - sim   # cosine distance, range [0, 2]

    same_w = wids.unsqueeze(0) == wids.unsqueeze(1)                  # [N, N]
    eye    = torch.eye(N, dtype=torch.bool, device=device)

    # --- Anchor-positive: same writer, both genuine, not self ----------------
    ap_mask = same_w & is_gen.unsqueeze(0) & is_gen.unsqueeze(1) & ~eye
    # invalid positions get -1 so they lose the max() (real distances are >= 0)
    d_ap, _ = dist.masked_fill(~ap_mask, -1.0).max(dim=1)   # hardest positive
    has_pos = ap_mask.any(dim=1)

    # --- Negative pool -------------------------------------------------------
    # Primary: same-writer skilled forgeries (key threat for a bank)
    an_primary   = same_w & is_forg.unsqueeze(0)
    # Secondary: different-writer genuines (easy negatives for regularisation;
    #   semi-hard mining naturally prefers primary since forgeries are closer)
    an_secondary = ~same_w & is_gen.unsqueeze(0)
    an_mask      = an_primary | an_secondary

    # --- Semi-hard selection -------------------------------------------------
    d_ap_exp = d_ap.unsqueeze(1).expand(N, N)
    sh_mask  = an_mask & (dist > d_ap_exp) & (dist < d_ap_exp + margin)
    has_sh   = sh_mask.any(dim=1)

    # Closest semi-hard negative; 2.0 is the max cosine distance so it loses min()
    d_neg_sh  = dist.masked_fill(~sh_mask, 2.0).min(dim=1).values
    # Closest valid negative (hardest) used when no semi-hard exists
    d_neg_hrd = dist.masked_fill(~an_mask, 2.0).min(dim=1).values

    d_neg   = torch.where(has_sh, d_neg_sh, d_neg_hrd)
    has_neg = an_mask.any(dim=1)

    # --- Loss ----------------------------------------------------------------
    anchor_mask = is_gen & has_pos & has_neg
    loss_active = F.relu(d_ap - d_neg + margin)[anchor_mask]
    n_triplets  = int(anchor_mask.sum().item())

    if n_triplets == 0:
        return embeddings.sum() * 0.0, 0   # zero loss with gradient

    return loss_active.mean(), n_triplets


# ---------------------------------------------------------------------------
# Loss-curve plot helper
# ---------------------------------------------------------------------------

def _save_loss_curve(epoch_losses: list, out_dir: str) -> None:
    epochs   = range(1, len(epoch_losses) + 1)
    best_idx = epoch_losses.index(min(epoch_losses))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, epoch_losses, marker="o", linewidth=2,
            markersize=5, color="steelblue", label="Training loss (triplet)")
    ax.scatter(best_idx + 1, epoch_losses[best_idx], color="red", zorder=5, s=80,
               label=f"Best: epoch {best_idx+1}  ({epoch_losses[best_idx]:.4f})")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Triplet Loss", fontsize=12)
    ax.set_title("Training Loss — Siamese Signature Verification", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5)

    path = os.path.join(out_dir, "loss_curve.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Loss curve saved -> {path}")


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(sanity: bool = False) -> None:
    """
    Full training pipeline.

    sanity=True: 1 epoch, 4 batches (~256 triplets) — pipeline smoke test.
    Healthy epoch-1 loss is roughly 0.05–0.30.  If it prints near zero
    (<0.01) the collapse-detection warning fires; do not proceed.
    """
    torch.manual_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device          : {device}")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # ----------------------------------------------------------------
    # Build disk cache (skips files that already exist)
    # ----------------------------------------------------------------
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cache_dir  = os.path.join(script_dir, CACHE_DIR)
    print("Building cache (skips existing files) …")
    build_cache(script_dir, cache_dir)

    # ----------------------------------------------------------------
    # Dataset + sampler + loader
    # ----------------------------------------------------------------
    dataset     = CEDARImageDataset(cache_dir, TRAIN_WRITERS)
    num_batches = 4               if sanity else NUM_BATCHES_PER_EPOCH
    epochs      = 1               if sanity else NUM_EPOCHS
    pk_seed     = SEED            if sanity else None   # fixed seed for reproducible sanity

    sampler = PKSampler(
        dataset,
        P=P, K_genuine=K_GENUINE, K_forgery=K_FORGERY,
        num_batches=num_batches,
        seed=pk_seed,
    )

    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
    )

    print(f"Train writers   : {len(TRAIN_WRITERS)}  ({len(dataset)} images)")
    print(f"Batch structure : P={P} writers x "
          f"(K_gen={K_GENUINE} + K_forg={K_FORGERY}) = "
          f"{P*(K_GENUINE+K_FORGERY)} imgs/batch")
    print(f"Batches/epoch   : {num_batches}  "
          f"(~{num_batches*P*K_GENUINE:,} triplets/epoch)")
    print(f"Epochs          : {epochs}")
    print(f"Margin          : {MARGIN}  (cosine distance)")
    print(f"LR              : {LEARNING_RATE}  "
          f"(halved after epoch {LR_DECAY_EPOCH})")
    if sanity:
        print("*** SANITY MODE — 1 epoch, 4 batches ***")
    print("-" * 55)

    # ----------------------------------------------------------------
    # Model / optimiser / scheduler
    # ----------------------------------------------------------------
    model     = SiameseNetwork(embedding_dim=EMBEDDING_DIM).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # Halve LR after LR_DECAY_EPOCH so later epochs do finer refinement
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[LR_DECAY_EPOCH], gamma=LR_DECAY_FACTOR
    )

    # ----------------------------------------------------------------
    # Training log
    # ----------------------------------------------------------------
    log_path = os.path.join(CHECKPOINT_DIR, "training_log.txt")
    log_file = open(log_path, "a", encoding="utf-8")
    run_start = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_file.write(
        f"\n--- Run started {run_start}  "
        f"(loss=triplet, margin={MARGIN}, lr={LEARNING_RATE}, "
        f"P={P}, K_gen={K_GENUINE}, K_forg={K_FORGERY}, "
        f"epochs={epochs}, batches/epoch={num_batches}"
        f"{'  SANITY' if sanity else ''}) ---\n"
    )
    log_file.flush()

    # ----------------------------------------------------------------
    # Training loop
    # ----------------------------------------------------------------
    epoch_losses = []
    best_loss    = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss     = 0.0
        running_triplets = 0
        n_valid_batches  = 0
        t0               = time.time()

        for imgs, writer_ids, is_genuine in loader:
            imgs = imgs.to(device)                     # [B, 1, 224, 224]
            embs = model._embed(imgs)                  # [B, 128], L2-normalised

            loss, n_trip = triplet_loss_semihard(
                embs,
                writer_ids.tolist(),
                is_genuine.tolist(),
                margin=MARGIN,
            )

            if n_trip == 0:
                continue   # no valid triplets in this batch (should be rare)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss     += loss.item()
            running_triplets += n_trip
            n_valid_batches  += 1

        if n_valid_batches == 0:
            print(f"WARNING epoch {epoch}: zero valid triplets — check PKSampler.")
            continue

        epoch_loss = running_loss / n_valid_batches
        duration   = time.time() - t0
        current_lr = optimizer.param_groups[0]["lr"]
        ts         = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        epoch_losses.append(epoch_loss)

        print(
            f"Epoch [{epoch:2d}/{epochs}]  "
            f"Loss: {epoch_loss:.6f}  "
            f"Triplets: {running_triplets:,}  "
            f"LR: {current_lr:.2e}  "
            f"({duration:.1f}s)  {ts}"
        )

        if epoch == 1:
            remaining = duration * (epochs - 1)
            print(f"  --> Est. remaining: {remaining/60:.1f} min "
                  f"({remaining/3600:.2f} hr)")
            # Collapse gate: triplet loss near zero after one epoch means the
            # semi-hard mining found no violating triplets — indicates collapse.
            if epoch_loss < 0.01:
                print(
                    f"\n  *** COLLAPSE WARNING: epoch-1 loss = {epoch_loss:.6f} "
                    f"is suspiciously near zero.  Expected 0.05–0.30 for a "
                    f"healthy run.  Do NOT proceed to full training without "
                    f"investigating. ***\n"
                )

        log_file.write(
            f"{ts}  Epoch {epoch:2d}/{epochs}  Loss: {epoch_loss:.6f}  "
            f"Triplets: {running_triplets}  Duration: {duration:.1f}s  "
            f"LR: {current_lr:.2e}\n"
        )
        log_file.flush()

        # Per-epoch checkpoint (includes optimiser state for resumability)
        ckpt_path = os.path.join(CHECKPOINT_DIR, f"siamese_epoch_{epoch:02d}.pt")
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

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(),
                       os.path.join(CHECKPOINT_DIR, "best.pt"))
            print(f"  --> New best saved  (loss: {best_loss:.6f})")

        scheduler.step()   # MultiStepLR: halves LR at epoch LR_DECAY_EPOCH

    # ----------------------------------------------------------------
    # Post-training outputs
    # ----------------------------------------------------------------
    torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "final.pt"))
    log_file.write(f"--- Training complete.  Best loss: {best_loss:.6f} ---\n")
    log_file.close()

    print("-" * 55)
    print(f"Training complete.")
    print(f"Best loss     : {best_loss:.6f}")
    print(f"Checkpoints   : ./{CHECKPOINT_DIR}/")
    print(f"Log           : {log_path}")

    _save_loss_curve(epoch_losses, CHECKPOINT_DIR)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Siamese signature-verification network (triplet loss)."
    )
    parser.add_argument(
        "--sanity", action="store_true",
        help="Smoke-test: 1 epoch, 4 batches.  Verify pipeline before full run.",
    )
    args = parser.parse_args()
    train(sanity=args.sanity)
