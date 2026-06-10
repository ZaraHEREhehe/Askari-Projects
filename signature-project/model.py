"""
model.py — Siamese Network for Signature Verification
=====================================================
A Siamese network passes two images through the *same* network (shared weights)
to produce embedding vectors, then compares them.  During training, pairs of
genuine / forged signatures are pushed apart or pulled together depending on
whether they match.

Architecture overview:
  Input (1-channel grayscale) ──► ResNet-18 backbone ──► 128-d L2-normalised embedding
                                    (shared weights)
  Input (1-channel grayscale) ──► ResNet-18 backbone ──► 128-d L2-normalised embedding

Both embedding vectors are returned from forward(); the loss function (e.g.
ContrastiveLoss or TripletLoss) is applied outside this file.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


# ---------------------------------------------------------------------------
# Helper: build the modified ResNet-18 backbone
# ---------------------------------------------------------------------------

def _build_backbone(embedding_dim: int = 128) -> nn.Module:
    """
    Load a pretrained ResNet-18 and make two modifications:

    1. Change the first convolution from 3-channel (RGB) → 1-channel (grayscale).
       Instead of discarding the pretrained weights, we *average* them across the
       colour dimension so the single grayscale channel still benefits from the
       ImageNet pretraining.

    2. Replace the final fully-connected classifier (1 000 ImageNet classes) with
       a linear layer that outputs `embedding_dim` features.  No activation is
       applied here; L2 normalisation is done in the forward pass.

    Parameters
    ----------
    embedding_dim : int
        Size of the output embedding vector (default: 128).

    Returns
    -------
    nn.Module
        Modified ResNet-18 ready to be used as a Siamese branch.
    """

    # ------------------------------------------------------------------ #
    # Step 1 – Load ResNet-18 with ImageNet pretrained weights.           #
    # weights=DEFAULT selects the best available pretrained checkpoint.   #
    # ------------------------------------------------------------------ #
    backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    # ------------------------------------------------------------------ #
    # Step 2 – Adapt the first conv layer for grayscale (1-channel) input.#
    #                                                                      #
    # Original layer: Conv2d(3, 64, kernel_size=7, stride=2, padding=3)   #
    # New layer     : Conv2d(1, 64, kernel_size=7, stride=2, padding=3)   #
    #                                                                      #
    # We copy the pretrained weights and collapse the 3 input channels    #
    # into 1 by averaging, preserving the learned spatial filters.        #
    # ------------------------------------------------------------------ #
    original_conv = backbone.conv1          # shape: [64, 3, 7, 7]

    new_conv = nn.Conv2d(
        in_channels=1,                      # grayscale: single channel
        out_channels=original_conv.out_channels,   # keep 64 output filters
        kernel_size=original_conv.kernel_size,
        stride=original_conv.stride,
        padding=original_conv.padding,
        bias=False,                         # ResNet-18 has no bias here
    )

    # Average pretrained RGB weights across the channel axis (dim=1):
    #   original weight shape: [64, 3, 7, 7]
    #   after mean(dim=1, keepdim=True): [64, 1, 7, 7]
    with torch.no_grad():
        new_conv.weight.copy_(
            original_conv.weight.mean(dim=1, keepdim=True)
        )

    backbone.conv1 = new_conv               # swap into the model

    # ------------------------------------------------------------------ #
    # Step 3 – Replace the classifier head with an embedding projector.   #
    #                                                                      #
    # ResNet-18's final layer is:  Linear(512, 1000)                      #
    # We replace it with:          Linear(512, embedding_dim)             #
    # ------------------------------------------------------------------ #
    in_features = backbone.fc.in_features   # 512 for ResNet-18
    backbone.fc = nn.Linear(in_features, embedding_dim, bias=True)

    return backbone


# ---------------------------------------------------------------------------
# Siamese Network
# ---------------------------------------------------------------------------

class SiameseNetwork(nn.Module):
    """
    Siamese Network that maps two signature images to L2-normalised embedding
    vectors in R^embedding_dim.

    Both branches share the *exact same* nn.Module instance (not just the same
    architecture with different weights), so every gradient update is applied
    identically to both branches.

    Usage
    -----
    model = SiameseNetwork()
    emb_a, emb_b = model(img_a, img_b)   # img_a/b: [B, 1, H, W] tensors
    """

    def __init__(self, embedding_dim: int = 128):
        """
        Parameters
        ----------
        embedding_dim : int
            Dimensionality of the output embedding (default: 128).
        """
        super().__init__()

        # One backbone instance → both inputs flow through the same weights.
        self.backbone = _build_backbone(embedding_dim)

    # ------------------------------------------------------------------
    def _embed(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pass a single batch of images through the backbone and L2-normalise.

        Parameters
        ----------
        x : torch.Tensor
            Shape [B, 1, H, W].  Pixel values should be normalised before
            calling (zero mean, unit std, or [0, 1] range).

        Returns
        -------
        torch.Tensor
            Shape [B, embedding_dim].  Each row has unit L2 norm.
        """
        features = self.backbone(x)         # [B, embedding_dim]

        # F.normalize divides each vector by its L2 norm so all embeddings
        # lie on the unit hypersphere.  This stabilises metric learning.
        embedding = F.normalize(features, p=2, dim=1)
        return embedding

    # ------------------------------------------------------------------
    def forward(
        self,
        img_a: torch.Tensor,
        img_b: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for a pair of signature images.

        Parameters
        ----------
        img_a, img_b : torch.Tensor
            Grayscale image batches, shape [B, 1, H, W].

        Returns
        -------
        emb_a, emb_b : tuple of torch.Tensor
            Both have shape [B, embedding_dim] and unit L2 norm.
            Pass these to your loss function (e.g. contrastive / triplet).
        """
        emb_a = self._embed(img_a)
        emb_b = self._embed(img_b)
        return emb_a, emb_b


# ---------------------------------------------------------------------------
# Quick sanity check — run this file directly to verify shapes
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    model = SiameseNetwork(embedding_dim=128)
    model.eval()

    # Simulate a batch of 4 grayscale signature images at 224×224
    dummy_a = torch.randn(4, 1, 224, 224)
    dummy_b = torch.randn(4, 1, 224, 224)

    with torch.no_grad():
        emb_a, emb_b = model(dummy_a, dummy_b)

    print(f"Embedding shape : {emb_a.shape}")          # expect [4, 128]
    print(f"L2 norm (emb_a) : {emb_a.norm(dim=1)}")   # expect all ~1.0

    total_params = sum(p.numel() for p in model.parameters())
    trainable    = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params    : {total_params:,}")
    print(f"Trainable params: {trainable:,}")
