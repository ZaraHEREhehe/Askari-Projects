"""
train.py — Dual-branch training with triplet + writer classification loss
=========================================================================
Trains the dual-branch SiameseNetwork on CEDAR writers 1-45.

Key upgrades from previous version
------------------------------------
Architecture : dual-branch ResNet-50 (semantic + detail-HF) with cross-attention
               local matching; 512-d embedding instead of 128-d.
Loss         : triplet_allpairs (margin 0.3) + writer CE classification (weight 0.3).
               The CE loss forces embeddings to be discriminative across writers,
               acting as a strong regularizer against within-class collapse.
Backbone     : stem + layer1 frozen; layer2-4 fine-tuned at 5e-5 (AdamW).
               New modules (local_attn, projector, classifier) at 1e-4.
Scheduler    : CosineAnnealingLR over 50 epochs → slow warm fade prevents
               the pretrained layers from destabilising early.
Grad clip    : max-norm 1.0 — stabilises the two-loss gradient mix.

Run
---
  python train.py            # full 50-epoch run
  python train.py --sanity   # 1 epoch, 4 batches — smoke test
"""

import os
import time
import argparse
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataset import (build_cache, build_cache_dataset2,
                     CombinedDataset, PKSampler,
                     TRAIN_WRITERS, D2_TRAIN_WRITERS)
from model   import SiameseNetwork


# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

EMBEDDING_DIM         = 512
NUM_EPOCHS            = 50
MARGIN                = 0.3    # cosine-distance triplet margin
CE_WEIGHT             = 0.3    # weight for writer classification loss

LR_BACKBONE           = 5e-5   # layer2-4 of both branches (pretrained, fine-tune slowly)
LR_NEW                = 1e-4   # local_attn + projector + classifier (trained from scratch)
WEIGHT_DECAY          = 1e-4
GRAD_CLIP_NORM        = 1.0

SEED                  = 42

P                     = 12    # writers per batch (reduced from 16 to fit dual ResNet-50)
K_GENUINE             = 4     # genuine images per writer per batch
K_FORGERY             = 4     # forgery images per writer per batch
NUM_BATCHES_PER_EPOCH = 125   # ~same epoch throughput as before (12*8*125 = 12000 imgs)

CHECKPOINT_DIR        = "checkpoints"
CACHE_DIR             = "cache"
NUM_WORKERS           = 0


# ---------------------------------------------------------------------------
# Triplet loss — all-pairs, same-writer forgeries only
# ---------------------------------------------------------------------------

def triplet_loss_allpairs(
    embeddings: torch.Tensor,
    writer_ids,
    is_genuine,
    margin: float = 0.3,
) -> tuple:
    """
    All-pairs triplet loss using same-writer skilled forgeries as negatives.
    For every genuine anchor, computes loss against every same-writer forgery.
    Uses the hardest positive (furthest same-writer genuine) per anchor.
    Returns (loss_scalar, n_active_triplets).
    """
    device  = embeddings.device
    N       = len(embeddings)

    wids    = torch.as_tensor(writer_ids, dtype=torch.long, device=device)
    is_gen  = torch.as_tensor(is_genuine, dtype=torch.bool, device=device)
    is_forg = ~is_gen

    sim  = torch.mm(embeddings, embeddings.t()).clamp(-1.0, 1.0)
    dist = 1.0 - sim

    same_w = wids.unsqueeze(0) == wids.unsqueeze(1)
    eye    = torch.eye(N, dtype=torch.bool, device=device)

    # Hardest positive per anchor: same writer, both genuine, not self
    ap_mask = same_w & is_gen.unsqueeze(0) & is_gen.unsqueeze(1) & ~eye
    d_ap = (dist * ap_mask.float()).sum(dim=1) / ap_mask.float().sum(dim=1).clamp(min=1)
    has_pos = ap_mask.any(dim=1)

    # Same-writer skilled forgeries only
    an_mask = same_w & is_forg.unsqueeze(0)
    has_neg = an_mask.any(dim=1)

    anchor_valid = is_gen & has_pos & has_neg
    d_ap_exp     = d_ap.unsqueeze(1).expand(N, N)
    pair_loss    = F.relu(d_ap_exp - dist + margin)
    valid        = anchor_valid.unsqueeze(1) & an_mask

    active     = pair_loss[valid]
    n_triplets = int(valid.sum().item())

    if n_triplets == 0:
        return embeddings.sum() * 0.0, 0

    return active.mean(), n_triplets


# ---------------------------------------------------------------------------
# Loss-curve plot
# ---------------------------------------------------------------------------

def _save_loss_curve(epoch_losses: list, out_dir: str) -> None:
    epochs   = range(1, len(epoch_losses) + 1)
    best_idx = epoch_losses.index(min(epoch_losses))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, epoch_losses, marker="o", linewidth=2,
            markersize=4, color="steelblue", label="Training loss (triplet + CE)")
    ax.scatter(best_idx + 1, epoch_losses[best_idx], color="red", zorder=5, s=80,
               label=f"Best: epoch {best_idx+1}  ({epoch_losses[best_idx]:.4f})")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title("Training Loss — Dual-Branch Signature Verification", fontsize=13)
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
    torch.manual_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device          : {device}")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # ----------------------------------------------------------------
    # Cache
    # ----------------------------------------------------------------
    script_dir   = os.path.dirname(os.path.abspath(__file__))
    cache_dir    = os.path.join(script_dir, CACHE_DIR)
    dataset2_dir = os.path.join(script_dir, "dataset2")
    print("Building CEDAR cache (skips existing files) …")
    build_cache(script_dir, cache_dir)
    print("Building dataset2 cache (skips existing files) …")
    build_cache_dataset2(dataset2_dir, cache_dir)

    # ----------------------------------------------------------------
    # Dataset / sampler / loader
    # ----------------------------------------------------------------
    dataset = CombinedDataset(
        cache_dir,
        cedar_writers=TRAIN_WRITERS,
        d2_writers=D2_TRAIN_WRITERS,   # 650 writers; 651-686 held out for testing
        augment=(not sanity),
    )
    # Build writer→label mapping for CE loss (IDs are non-contiguous with offset)
    all_train_writers = sorted(dataset.genuine_by_writer.keys())
    num_train_writers = len(all_train_writers)
    wid_to_label      = {wid: i for i, wid in enumerate(all_train_writers)}

    num_batches = 4               if sanity else NUM_BATCHES_PER_EPOCH
    epochs      = 1               if sanity else NUM_EPOCHS
    pk_seed     = SEED            if sanity else None

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

    print(f"Train writers   : {num_train_writers}  "
          f"(CEDAR {len(TRAIN_WRITERS)} + Dataset2 {len(D2_TRAIN_WRITERS)})  "
          f"({len(dataset)} images)")
    print(f"Batch structure : P={P} writers x "
          f"(K_gen={K_GENUINE} + K_forg={K_FORGERY}) = "
          f"{P*(K_GENUINE+K_FORGERY)} imgs/batch")
    print(f"Batches/epoch   : {num_batches}")
    print(f"Epochs          : {epochs}")
    print(f"Margin          : {MARGIN}  (cosine dist)   CE weight: {CE_WEIGHT}")
    print(f"Augmentation    : {'OFF (sanity)' if sanity else 'ON'}")
    if sanity:
        print("*** SANITY MODE — 1 epoch, 4 batches ***")
    print("-" * 55)

    # ----------------------------------------------------------------
    # Model
    # ----------------------------------------------------------------
    model = SiameseNetwork(embedding_dim=EMBEDDING_DIM).to(device)

    # Freeze stem (first conv + bn) and layer1 in both branches.
    # These low-level texture detectors are already well-pretrained;
    # updating them risks destabilising the HF detail channel in the
    # early training phase.
    frozen_prefixes = (
        "sem_branch.stem", "sem_branch.layer1",
        "det_branch.stem", "det_branch.layer1",
    )
    for name, param in model.named_parameters():
        if name.startswith(frozen_prefixes):
            param.requires_grad = False

    n_total     = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {n_trainable:,} / {n_total:,}  "
          f"(stem+layer1 frozen in both branches)")

    # ----------------------------------------------------------------
    # Writer classification head (training only — not used at inference)
    # ----------------------------------------------------------------
    classifier = nn.Linear(EMBEDDING_DIM, num_train_writers).to(device)

    # ----------------------------------------------------------------
    # Optimizer — differential LR
    # ----------------------------------------------------------------
    backbone_params = []   # pretrained layers 2-4 (fine-tune slowly)
    new_params      = []   # local_attn + projector (train from scratch)

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith(("sem_branch.layer", "det_branch.layer")):
            backbone_params.append(param)
        else:
            new_params.append(param)

    optimizer = torch.optim.AdamW([
        {"params": backbone_params,           "lr": LR_BACKBONE},
        {"params": new_params,                "lr": LR_NEW},
        {"params": classifier.parameters(),   "lr": LR_NEW},
    ], weight_decay=WEIGHT_DECAY)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )

    print(f"LR backbone(layer2-4): {LR_BACKBONE}   LR new modules: {LR_NEW}")
    print(f"Scheduler: CosineAnnealingLR  T_max={epochs}  eta_min=1e-6")
    print("-" * 55)

    # ----------------------------------------------------------------
    # Training log
    # ----------------------------------------------------------------
    log_path = os.path.join(CHECKPOINT_DIR, "training_log.txt")
    log_file = open(log_path, "a", encoding="utf-8")
    run_start = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_file.write(
        f"\n--- Run started {run_start}  "
        f"(dual-branch ResNet50, emb={EMBEDDING_DIM}, margin={MARGIN}, "
        f"ce_weight={CE_WEIGHT}, lr_backbone={LR_BACKBONE}, lr_new={LR_NEW}, "
        f"wd={WEIGHT_DECAY}, P={P}, K_gen={K_GENUINE}, K_forg={K_FORGERY}, "
        f"epochs={epochs}, batches/epoch={num_batches}, "
        f"writers={num_train_writers} (CEDAR {len(TRAIN_WRITERS)} + D2 {len(D2_TRAIN_WRITERS)})"
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
        classifier.train()
        running_loss     = 0.0
        running_triplets = 0
        n_valid_batches  = 0
        t0               = time.time()

        for imgs, writer_ids, is_genuine in loader:
            imgs = imgs.to(device)

            embs = model._embed(imgs)   # [B, 512]

            # --- Triplet loss ---
            trip_loss, n_trip = triplet_loss_allpairs(
                embs,
                writer_ids.tolist(),
                is_genuine.tolist(),
                margin=MARGIN,
            )

            # --- Writer classification loss (genuine images only) ---
            gen_mask = is_genuine.bool()
            if gen_mask.sum() > 0:
                gen_embs = embs[gen_mask]
                gen_wids = torch.tensor(
                    [wid_to_label[int(w)] for w in writer_ids[gen_mask]],
                    dtype=torch.long, device=device,
                )
                logits   = classifier(gen_embs)
                ce_loss  = F.cross_entropy(logits, gen_wids)
            else:
                ce_loss = embs.sum() * 0.0

            if n_trip == 0:
                continue

            total_loss = trip_loss + CE_WEIGHT * ce_loss

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(classifier.parameters()),
                GRAD_CLIP_NORM,
            )
            optimizer.step()

            running_loss     += total_loss.item()
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
            print(f"  --> Est. remaining: {remaining/60:.1f} min  "
                  f"({remaining/3600:.2f} hr)")
            if epoch_loss < 0.005:
                print(
                    f"\n  *** COLLAPSE WARNING: epoch-1 loss = {epoch_loss:.6f}.  "
                    f"Expected 0.05–0.40 for a healthy dual-branch run. ***\n"
                )

        log_file.write(
            f"{ts}  Epoch {epoch:2d}/{epochs}  Loss: {epoch_loss:.6f}  "
            f"Triplets: {running_triplets}  Duration: {duration:.1f}s  "
            f"LR: {current_lr:.2e}\n"
        )
        log_file.flush()

        # Full checkpoint (includes optimizer + classifier for resumability)
        ckpt_path = os.path.join(CHECKPOINT_DIR, f"siamese_epoch_{epoch:02d}.pt")
        torch.save(
            {
                "epoch"                : epoch,
                "model_state_dict"     : model.state_dict(),
                "classifier_state_dict": classifier.state_dict(),
                "optim_state_dict"     : optimizer.state_dict(),
                "loss"                 : epoch_loss,
                "margin"               : MARGIN,
                "embedding_dim"        : EMBEDDING_DIM,
            },
            ckpt_path,
        )

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(),
                       os.path.join(CHECKPOINT_DIR, "best.pt"))
            print(f"  --> New best saved  (loss: {best_loss:.6f})")

        scheduler.step()

    # ----------------------------------------------------------------
    # Post-training
    # ----------------------------------------------------------------
    torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "final.pt"))
    log_file.write(f"--- Training complete.  Best loss: {best_loss:.6f} ---\n")
    log_file.close()

    print("-" * 55)
    print(f"Training complete.  Best loss: {best_loss:.6f}")
    print(f"Checkpoints: ./{CHECKPOINT_DIR}/")

    _save_loss_curve(epoch_losses, CHECKPOINT_DIR)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train dual-branch signature-verification network."
    )
    parser.add_argument(
        "--sanity", action="store_true",
        help="Smoke-test: 1 epoch, 4 batches.",
    )
    args = parser.parse_args()
    train(sanity=args.sanity)
