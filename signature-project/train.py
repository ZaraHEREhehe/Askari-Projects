"""
train.py — Dual-branch training with ArcFace + triplet loss
============================================================
Trains on CEDAR writers 1-45 + Dataset2 writers 1-650.

What changed from v1 (and why)
--------------------------------
ArcFace loss   : Replaces plain cross-entropy for writer classification.
                 Adds an angular margin to the target class angle before softmax,
                 forcing embeddings into tighter, more separated clusters per writer.
                 This is the single biggest improvement for metric learning tasks.

Stratified     : StratifiedPKSampler guarantees 4 CEDAR + 8 Dataset2 writers
sampler          per batch instead of random uniform.  Fixes the 14:1 imbalance
                 that was under-exposing CEDAR patterns.

100 epochs     : Larger combined dataset needs more iterations to converge.

Run
---
  python train.py            # full 100-epoch run (~4-5 hrs on T4 GPU)
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
                     CombinedDataset, StratifiedPKSampler,
                     TRAIN_WRITERS, D2_TRAIN_WRITERS)
from model   import SiameseNetwork


# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

EMBEDDING_DIM         = 512
NUM_EPOCHS            = 100    # was 50 — larger dataset needs more iterations

MARGIN                = 0.3    # cosine-distance triplet margin

# ArcFace replaces plain cross-entropy for writer classification.
# scale=30 is the standard for verification tasks; margin=0.35 ≈ 20 degrees.
ARCFACE_SCALE         = 30.0
ARCFACE_MARGIN        = 0.35
ARCFACE_WEIGHT        = 0.3    # weight of ArcFace term in the total loss

LR_BACKBONE           = 5e-5   # layer2-4 of both branches (pretrained, fine-tune slowly)
LR_NEW                = 1e-4   # local_attn + projector + arcface (trained from scratch)
WEIGHT_DECAY          = 1e-4
GRAD_CLIP_NORM        = 1.0

SEED                  = 42

# Stratified batch: 4 CEDAR + 8 Dataset2 writers per batch.
# Keeps the total at 12 writers but gives CEDAR ~5x more exposure than
# random sampling would (natural proportion was 1 CEDAR per 14 D2 writers).
P_CEDAR               = 4
P_D2                  = 8
K_GENUINE             = 4      # genuine images per writer per batch
K_FORGERY             = 4      # forgery images per writer per batch
NUM_BATCHES_PER_EPOCH = 125

CHECKPOINT_DIR        = "checkpoints"
CACHE_DIR             = "cache"
NUM_WORKERS           = 0


# ---------------------------------------------------------------------------
# ArcFace loss
# ---------------------------------------------------------------------------

class ArcFaceLoss(nn.Module):
    """
    Additive Angular Margin loss for writer classification.

    How it works:
      1. L2-normalise both the embedding vector and each class weight vector
         so all lie on the unit sphere — the angle between them is meaningful.
      2. Compute the angle θ between the embedding and its correct class weight.
      3. Add the angular margin m to θ for the target class only, then scale
         by s and apply softmax cross-entropy.

    Effect: embeddings are pushed further from every decision boundary,
    creating tighter intra-class clusters and wider inter-class gaps compared
    to plain cross-entropy.  This is the primary improvement over v1.

    Parameters
    ----------
    num_classes   : number of training writers (CEDAR + Dataset2 combined)
    embedding_dim : size of L2-normalised output vector (512)
    scale         : logit multiplier s — controls softmax sharpness (default 30)
    margin        : angular margin m in radians (default 0.35 ≈ 20°)
    """

    def __init__(
        self,
        num_classes:   int,
        embedding_dim: int,
        scale:         float = 30.0,
        margin:        float = 0.35,
    ):
        super().__init__()
        self.scale  = scale
        self.margin = margin
        # Learnable class weight matrix — each row is a class centre on the unit sphere
        self.weight = nn.Parameter(torch.empty(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # Normalise class weights to unit sphere (embeddings already normalised by model)
        w      = F.normalize(self.weight, p=2, dim=1)
        cosine = torch.mm(embeddings, w.t()).clamp(-1 + 1e-7, 1 - 1e-7)

        # Add angular margin m to target class angle, leave other classes unchanged
        theta   = torch.acos(cosine)
        one_hot = torch.zeros_like(cosine).scatter_(1, labels.unsqueeze(1), 1.0)
        output  = torch.cos(theta + self.margin * one_hot) * self.scale

        return F.cross_entropy(output, labels)


# ---------------------------------------------------------------------------
# Triplet loss — all-pairs, same-writer forgeries as negatives
# ---------------------------------------------------------------------------

def triplet_loss_allpairs(
    embeddings: torch.Tensor,
    writer_ids,
    is_genuine,
    margin: float = 0.3,
) -> tuple:
    """
    All-pairs triplet loss using same-writer skilled forgeries as negatives.
    For every genuine anchor:
      positive  = hardest same-writer genuine (furthest cosine distance)
      negatives = all same-writer forgeries

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

    # Hardest positive: same writer, both genuine, not self
    ap_mask = same_w & is_gen.unsqueeze(0) & is_gen.unsqueeze(1) & ~eye
    d_ap    = (dist * ap_mask.float()).sum(1) / ap_mask.float().sum(1).clamp(min=1)
    has_pos = ap_mask.any(1)

    # Negatives: all same-writer forgeries
    an_mask = same_w & is_forg.unsqueeze(0)
    has_neg = an_mask.any(1)

    anchor_valid = is_gen & has_pos & has_neg
    pair_loss    = F.relu(d_ap.unsqueeze(1).expand(N, N) - dist + margin)
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
    ax.plot(epochs, epoch_losses, marker="o", linewidth=2, markersize=3,
            color="steelblue", label="Training loss (triplet + ArcFace)")
    ax.scatter(best_idx + 1, epoch_losses[best_idx], color="red",
               zorder=5, s=80,
               label=f"Best: epoch {best_idx+1}  ({epoch_losses[best_idx]:.4f})")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss",  fontsize=12)
    ax.set_title("Training Loss — Dual-Branch Signature Verification", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5)

    path = os.path.join(out_dir, "loss_curve.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Loss curve saved → {path}")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(sanity: bool = False) -> None:
    torch.manual_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device          : {device}")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # ── Cache ────────────────────────────────────────────────────────────
    script_dir   = os.path.dirname(os.path.abspath(__file__))
    cache_dir    = os.path.join(script_dir, CACHE_DIR)
    dataset2_dir = os.path.join(script_dir, "dataset2")

    print("Building CEDAR cache (skips existing) …")
    build_cache(script_dir, cache_dir)
    print("Building Dataset2 cache (skips existing) …")
    build_cache_dataset2(dataset2_dir, cache_dir)

    # ── Dataset ───────────────────────────────────────────────────────────
    dataset = CombinedDataset(
        cache_dir,
        cedar_writers=TRAIN_WRITERS,
        d2_writers=D2_TRAIN_WRITERS,   # 650 writers; 651-686 held out for testing
        augment=(not sanity),
    )

    # Build writer → integer label map for ArcFace (IDs are non-contiguous after offset)
    all_train_writers = sorted(dataset.genuine_by_writer.keys())
    num_train_writers = len(all_train_writers)
    wid_to_label      = {wid: i for i, wid in enumerate(all_train_writers)}

    num_batches = 4      if sanity else NUM_BATCHES_PER_EPOCH
    epochs      = 1      if sanity else NUM_EPOCHS
    pk_seed     = SEED   if sanity else None

    # ── Stratified sampler ────────────────────────────────────────────────
    sampler = StratifiedPKSampler(
        dataset,
        P_cedar=P_CEDAR, P_d2=P_D2,
        K_genuine=K_GENUINE, K_forgery=K_FORGERY,
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
    print(f"Batch structure : {P_CEDAR} CEDAR + {P_D2} D2 writers × "
          f"(K_gen={K_GENUINE} + K_forg={K_FORGERY}) = "
          f"{(P_CEDAR + P_D2) * (K_GENUINE + K_FORGERY)} imgs/batch")
    print(f"Batches/epoch   : {num_batches}    Epochs: {epochs}")
    print(f"Triplet margin  : {MARGIN}   "
          f"ArcFace: scale={ARCFACE_SCALE}  margin={ARCFACE_MARGIN}rad  "
          f"weight={ARCFACE_WEIGHT}")
    print(f"Augmentation    : {'OFF (sanity)' if sanity else 'ON'}")
    if sanity:
        print("*** SANITY MODE — 1 epoch, 4 batches ***")
    print("-" * 60)

    # ── Model ────────────────────────────────────────────────────────────
    model = SiameseNetwork(embedding_dim=EMBEDDING_DIM).to(device)

    # Freeze stem and layer1 in both branches (low-level pretrained features —
    # updating them early risks destabilising the high-frequency detail channel)
    frozen_prefixes = (
        "sem_branch.stem", "sem_branch.layer1",
        "det_branch.stem", "det_branch.layer1",
    )
    for name, param in model.named_parameters():
        if name.startswith(frozen_prefixes):
            param.requires_grad = False

    n_total     = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {n_trainable:,} / {n_total:,}  (stem+layer1 frozen)")

    # ── ArcFace head ─────────────────────────────────────────────────────
    arcface = ArcFaceLoss(
        num_train_writers, EMBEDDING_DIM, ARCFACE_SCALE, ARCFACE_MARGIN
    ).to(device)

    # ── Optimizer — differential learning rates ───────────────────────────
    backbone_params, new_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith(("sem_branch.layer", "det_branch.layer")):
            backbone_params.append(param)
        else:
            new_params.append(param)

    optimizer = torch.optim.AdamW([
        {"params": backbone_params,      "lr": LR_BACKBONE},  # ResNet layers 2-4
        {"params": new_params,           "lr": LR_NEW},       # attention + projector
        {"params": arcface.parameters(), "lr": LR_NEW},       # ArcFace class weights
    ], weight_decay=WEIGHT_DECAY)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )

    print(f"LR backbone(layer2-4): {LR_BACKBONE}   LR new modules: {LR_NEW}")
    print(f"Scheduler: CosineAnnealingLR  T_max={epochs}  eta_min=1e-6")
    print("-" * 60)

    # ── Training log ─────────────────────────────────────────────────────
    log_path  = os.path.join(CHECKPOINT_DIR, "training_log.txt")
    log_file  = open(log_path, "a", encoding="utf-8")
    run_start = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_file.write(
        f"\n--- Run started {run_start}  "
        f"(dual-branch ResNet50, emb={EMBEDDING_DIM}, "
        f"triplet margin={MARGIN}, "
        f"arcface scale={ARCFACE_SCALE} margin={ARCFACE_MARGIN} weight={ARCFACE_WEIGHT}, "
        f"lr_backbone={LR_BACKBONE}, lr_new={LR_NEW}, wd={WEIGHT_DECAY}, "
        f"P_cedar={P_CEDAR}, P_d2={P_D2}, K_gen={K_GENUINE}, K_forg={K_FORGERY}, "
        f"epochs={epochs}, batches/epoch={num_batches}, "
        f"writers={num_train_writers} "
        f"(CEDAR {len(TRAIN_WRITERS)} + D2 {len(D2_TRAIN_WRITERS)})"
        f"{'  SANITY' if sanity else ''}) ---\n"
    )
    log_file.flush()

    # ── Training loop ────────────────────────────────────────────────────
    epoch_losses = []
    best_loss    = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        arcface.train()
        running_loss     = 0.0
        running_triplets = 0
        n_valid_batches  = 0
        t0               = time.time()

        for imgs, writer_ids, is_genuine in loader:
            imgs = imgs.to(device)
            embs = model._embed(imgs)   # [B, 512], L2-normalised

            # Triplet loss — metric learning between genuine / forgery pairs
            trip_loss, n_trip = triplet_loss_allpairs(
                embs, writer_ids.tolist(), is_genuine.tolist(), margin=MARGIN,
            )

            # ArcFace loss — writer discrimination (genuine images only)
            # Forgeries are excluded because they don't have a stable class identity
            gen_mask = is_genuine.bool()
            if gen_mask.sum() > 0:
                gen_embs = embs[gen_mask]
                gen_wids = torch.tensor(
                    [wid_to_label[int(w)] for w in writer_ids[gen_mask]],
                    dtype=torch.long, device=device,
                )
                arc_loss = arcface(gen_embs, gen_wids)
            else:
                arc_loss = embs.sum() * 0.0

            if n_trip == 0:
                continue

            total_loss = trip_loss + ARCFACE_WEIGHT * arc_loss

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(arcface.parameters()),
                GRAD_CLIP_NORM,
            )
            optimizer.step()

            running_loss     += total_loss.item()
            running_triplets += n_trip
            n_valid_batches  += 1

        if n_valid_batches == 0:
            print(f"WARNING epoch {epoch}: zero valid triplets — check sampler.")
            continue

        epoch_loss = running_loss / n_valid_batches
        duration   = time.time() - t0
        current_lr = optimizer.param_groups[0]["lr"]
        ts         = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        epoch_losses.append(epoch_loss)

        print(
            f"Epoch [{epoch:3d}/{epochs}]  "
            f"Loss: {epoch_loss:.6f}  "
            f"Triplets: {running_triplets:,}  "
            f"LR: {current_lr:.2e}  "
            f"({duration:.1f}s)  {ts}"
        )

        if epoch == 1:
            remaining = duration * (epochs - 1)
            print(f"  --> Est. remaining: {remaining/60:.1f} min  "
                  f"({remaining/3600:.2f} hr)")
            # With ArcFace(scale=30) the combined loss starts around 3-6.
            # A value below 0.1 after epoch 1 means triplet collapsed.
            if epoch_loss < 0.1:
                print(
                    f"\n  *** COLLAPSE WARNING: epoch-1 loss = {epoch_loss:.6f}. "
                    f"Expected 3.0–6.0 for a healthy ArcFace run. ***\n"
                )

        log_file.write(
            f"{ts}  Epoch {epoch:3d}/{epochs}  Loss: {epoch_loss:.6f}  "
            f"Triplets: {running_triplets}  Duration: {duration:.1f}s  "
            f"LR: {current_lr:.2e}\n"
        )
        log_file.flush()

        # Full checkpoint (model + arcface + optimizer — allows resuming)
        ckpt_path = os.path.join(CHECKPOINT_DIR, f"siamese_epoch_{epoch:03d}.pt")
        torch.save(
            {
                "epoch"              : epoch,
                "model_state_dict"   : model.state_dict(),
                "arcface_state_dict" : arcface.state_dict(),
                "optim_state_dict"   : optimizer.state_dict(),
                "loss"               : epoch_loss,
                "margin"             : MARGIN,
                "arcface_scale"      : ARCFACE_SCALE,
                "arcface_margin"     : ARCFACE_MARGIN,
                "embedding_dim"      : EMBEDDING_DIM,
            },
            ckpt_path,
        )

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "best.pt"))
            print(f"  --> New best saved  (loss: {best_loss:.6f})")

        scheduler.step()

    # ── Post-training ────────────────────────────────────────────────────
    torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "final.pt"))
    log_file.write(f"--- Training complete.  Best loss: {best_loss:.6f} ---\n")
    log_file.close()

    print("-" * 60)
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
