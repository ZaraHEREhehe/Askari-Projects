"""
model.py — Dual-Branch Signature Verification Network
=====================================================
Two branches extract complementary features from each signature image:

  Semantic branch : ResNet-50 on raw grayscale     → overall shape / structure
  Detail branch   : ResNet-50 on [grayscale + HF]  → fine pen strokes / pressure

"HF" = a fixed Laplacian channel that amplifies high-frequency edges, making
subtle stroke differences between genuine and forged signatures visible to the
detail branch while the semantic branch focuses on global shape.

Cross-attention at layer-3 feature maps (14×14 spatial) lets the semantic
branch attend to specific local regions of the detail branch — this is the
local structure matching described in DetailSemNet (ECCV 2024).

Final embedding: concat(sem_global, det_global, fused_local) → 5120-d
                 → projector → 512-d L2-normalised embedding.

Interface is backward-compatible:
  model(img_a, img_b)  →  (emb_a, emb_b)  both [B, embedding_dim]
  model._embed(x)      →  emb              [B, embedding_dim]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


# ---------------------------------------------------------------------------
# 1. High-frequency extractor (fixed Laplacian — no learned params)
# ---------------------------------------------------------------------------

class _HighFreqExtractor(nn.Module):
    """
    Appends a Laplacian high-pass channel to the input image.
    The Laplacian sharpens edges and stroke boundaries, which is where
    skilled forgeries deviate from the genuine signature.

    Input  : [B, 1, H, W]
    Output : [B, 2, H, W]  (original grayscale + high-freq residual)
    """

    def __init__(self):
        super().__init__()
        kernel = torch.tensor(
            [[0, -1,  0],
             [-1,  4, -1],
             [0, -1,  0]], dtype=torch.float32
        ).view(1, 1, 3, 3)
        self.register_buffer("kernel", kernel)   # fixed, not a parameter

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hf = F.conv2d(x, self.kernel, padding=1)
        return torch.cat([x, hf], dim=1)


# ---------------------------------------------------------------------------
# 2. ResNet-50 branch with intermediate feature access
# ---------------------------------------------------------------------------

class _ResNet50Branch(nn.Module):
    """
    ResNet-50 backbone adapted for grayscale or 2-channel input.
    Returns both layer-3 feature maps (for local cross-attention) and the
    final global-pooled vector (for global similarity).

    layer3 output : [B, 1024, 14, 14]
    global output : [B, 2048]
    """

    def __init__(self, in_channels: int):
        super().__init__()
        base = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        new_conv = nn.Conv2d(in_channels, 64, kernel_size=7,
                             stride=2, padding=3, bias=False)
        with torch.no_grad():
            avg_w = base.conv1.weight.mean(dim=1, keepdim=True)   # [64, 1, 7, 7]
            for c in range(in_channels):
                new_conv.weight[:, c:c+1].copy_(avg_w)

        self.stem    = nn.Sequential(new_conv, base.bn1, base.relu, base.maxpool)
        self.layer1  = base.layer1
        self.layer2  = base.layer2
        self.layer3  = base.layer3    # [B, 1024, 14, 14]
        self.layer4  = base.layer4    # [B, 2048,  7,  7]
        self.avgpool = base.avgpool   # [B, 2048,  1,  1]

    def forward(self, x: torch.Tensor):
        x  = self.stem(x)
        x  = self.layer1(x)
        x  = self.layer2(x)
        l3 = self.layer3(x)
        g  = self.avgpool(self.layer4(l3)).flatten(1)   # [B, 2048]
        return l3, g


# ---------------------------------------------------------------------------
# 3. Cross-attention local structure matching
# ---------------------------------------------------------------------------

class _LocalAttentionFusion(nn.Module):
    """
    Multi-head cross-attention: semantic features (query) attend to detail
    features (key/value) at the spatial level.  This allows the model to
    compare local stroke regions between the two branches rather than just
    comparing global statistics.

    Spatial maps are pooled to 7×7 = 49 tokens before attention for memory
    efficiency, then the attended output is upsampled and added as a residual
    to the full-resolution semantic map.

    Input : sem [B, 1024, 14, 14],  det [B, 1024, 14, 14]
    Output: fused [B, 1024, 14, 14]
    """

    def __init__(self, channels: int = 1024, heads: int = 4, pool_to: int = 7):
        super().__init__()
        assert channels % heads == 0
        self.heads    = heads
        self.head_dim = channels // heads
        self.scale    = self.head_dim ** -0.5
        self.pool     = nn.AdaptiveAvgPool2d(pool_to)
        self.q_proj   = nn.Conv2d(channels, channels, 1)
        self.k_proj   = nn.Conv2d(channels, channels, 1)
        self.v_proj   = nn.Conv2d(channels, channels, 1)
        self.out_proj = nn.Conv2d(channels, channels, 1)
        self.norm     = nn.GroupNorm(8, channels)

    def forward(self, sem: torch.Tensor, det: torch.Tensor) -> torch.Tensor:
        sem_s = self.pool(sem)   # [B, C, 7, 7]
        det_s = self.pool(det)
        B, C, H, W = sem_s.shape
        N  = H * W
        h  = self.heads
        hd = self.head_dim

        Q = self.q_proj(sem_s).view(B, h, hd, N).permute(0, 1, 3, 2)  # [B, h, N, hd]
        K = self.k_proj(det_s).view(B, h, hd, N)                       # [B, h, hd, N]
        V = self.v_proj(det_s).view(B, h, hd, N).permute(0, 1, 3, 2)  # [B, h, N, hd]

        attn = torch.matmul(Q, K) * self.scale   # [B, h, N, N]
        attn = F.softmax(attn, dim=-1)
        out  = torch.matmul(attn, V)             # [B, h, N, hd]
        out  = out.permute(0, 1, 3, 2).reshape(B, C, H, W)
        out  = self.out_proj(out)

        out_up = F.interpolate(out, size=sem.shape[2:],
                               mode="bilinear", align_corners=False)
        return self.norm(sem + out_up)


# ---------------------------------------------------------------------------
# 4. Main model
# ---------------------------------------------------------------------------

class SiameseNetwork(nn.Module):
    """
    Dual-branch Siamese network for signature verification.

    Each input goes through:
      1. Semantic branch  (ResNet-50, 1-ch)     → l3_sem [B,1024,14,14], g_sem [B,2048]
      2. Detail branch    (ResNet-50, 2-ch HF)  → l3_det [B,1024,14,14], g_det [B,2048]
      3. Cross-attention  (sem attends to det)  → fused  [B,1024,14,14]
      4. GAP on fused                            → local_g [B,1024]
      5. Concat [g_sem, g_det, local_g]          → [B, 5120]
      6. Projector                               → [B, embedding_dim], L2-normalised

    Usage
    -----
    model = SiameseNetwork(embedding_dim=512)
    emb_a, emb_b = model(img_a, img_b)   # [B, 1, 224, 224] each
    emb = model._embed(img)
    """

    def __init__(self, embedding_dim: int = 512):
        super().__init__()
        self.embedding_dim = embedding_dim

        self.hf         = _HighFreqExtractor()
        self.sem_branch = _ResNet50Branch(in_channels=1)
        self.det_branch = _ResNet50Branch(in_channels=2)
        self.local_attn = _LocalAttentionFusion(channels=1024, heads=4, pool_to=7)
        self.gap        = nn.AdaptiveAvgPool2d(1)

        self.projector  = nn.Sequential(
            nn.Linear(2048 + 2048 + 1024, 1024),
            nn.BatchNorm1d(1024),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(1024, embedding_dim),
        )

    def _embed(self, x: torch.Tensor) -> torch.Tensor:
        x_hf          = self.hf(x)                          # [B, 2, H, W]
        sem_l3, sem_g = self.sem_branch(x)                  # l3:[B,1024,14,14], g:[B,2048]
        det_l3, det_g = self.det_branch(x_hf)               # l3:[B,1024,14,14], g:[B,2048]
        fused_l3      = self.local_attn(sem_l3, det_l3)     # [B, 1024, 14, 14]
        local_g       = self.gap(fused_l3).flatten(1)       # [B, 1024]
        combined      = torch.cat([sem_g, det_g, local_g], dim=1)   # [B, 5120]
        return F.normalize(self.projector(combined), p=2, dim=1)

    def forward(
        self,
        img_a: torch.Tensor,
        img_b: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._embed(img_a), self._embed(img_b)


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    model = SiameseNetwork(embedding_dim=512)
    model.eval()

    dummy_a = torch.randn(2, 1, 224, 224)
    dummy_b = torch.randn(2, 1, 224, 224)

    with torch.no_grad():
        emb_a, emb_b = model(dummy_a, dummy_b)

    print(f"Embedding shape : {emb_a.shape}")
    print(f"L2 norm (emb_a) : {emb_a.norm(dim=1)}")

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params    : {total:,}")
    print(f"Trainable params: {trainable:,}")
