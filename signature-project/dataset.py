"""
dataset.py
==========
PyTorch Dataset for the CEDAR offline signature-verification dataset.

CEDAR folder layout (both directories sit next to this script):
  full_org/   – genuine (original) signatures  → original_{writer}_{n}.png
  full_forg/  – skilled forgeries              → forgeries_{writer}_{n}.png

Dataset statistics:
  • 55 writers total
  • 24 genuine  images per writer (n = 1 … 24)
  • 24 forgery  images per writer (n = 1 … 24)

Writer split  (test writers NEVER appear in training):
  Training : writers  1 – 45   (45 writers)
  Testing  : writers 46 – 55   (10 writers)

Every sample returned by __getitem__ is a PAIR of preprocessed images plus a
binary label describing the relationship between those two images:
  label = 1  →  both images are genuine signatures of the SAME writer
  label = 0  →  one genuine + one forgery of the SAME writer

All pairs are built once at construction time, so training is fully
reproducible for a fixed random seed.  Positive (label 1) and negative
(label 0) pairs are kept at a 1 : 1 ratio so the model sees a balanced
training signal.
"""

# ---------------------------------------------------------------------------
# Standard-library imports
# ---------------------------------------------------------------------------
import os           # file-path operations
import itertools    # combinatorial helpers (combinations)
import random       # pair sampling with a local RNG

# ---------------------------------------------------------------------------
# Third-party imports
# ---------------------------------------------------------------------------
import numpy as np          # NumPy array manipulation
import torch                # PyTorch tensors
from torch.utils.data import Dataset  # base class we must subclass

# ---------------------------------------------------------------------------
# Local import – the preprocessing pipeline from preprocess.py
# ---------------------------------------------------------------------------
from preprocess import preprocess_signature
# preprocess_signature(path) → uint8 NumPy array, shape (224, 224)
# Values:  0 = ink (dark),  255 = background (white)


# ---------------------------------------------------------------------------
# Constants that describe the CEDAR dataset
# ---------------------------------------------------------------------------

TOTAL_WRITERS = 55      # writers are numbered 1 … 55
SAMPLES_EACH  = 24      # genuine count = forgery count = 24 per writer

# Writer IDs for each split – these two lists must not overlap.
TRAIN_WRITERS = list(range(1, 46))    # [1, 2, …, 45]
TEST_WRITERS  = list(range(46, 56))   # [46, 47, …, 55]


# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------

class CEDARSignatureDataset(Dataset):
    """
    Pair-based PyTorch Dataset for CEDAR signature verification.

    Each item is a triple  (img1, img2, label)  where:
      • img1, img2 are preprocessed signature images as float32 tensors
        of shape (1, 224, 224)  — channel-first, values in [0.0, 1.0].
      • label is a scalar int64 tensor:  1 = genuine pair,  0 = forgery pair.

    Parameters
    ----------
    root_dir : str
        Directory that contains the  full_org/  and  full_forg/  sub-folders.
        Typically the folder where this script lives.
    split : str
        "train"  uses writers  1-45.
        "test"   uses writers 46-55.
    seed : int
        Random seed for sampling the forgery pairs.  Fix this to get the same
        pairs across runs; change it to experiment with different splits.
    """

    def __init__(self, root_dir: str, split: str = "train", seed: int = 42):

        # ----------------------------------------------------------------
        # Validate arguments up-front so errors are easy to understand.
        # ----------------------------------------------------------------
        if split not in ("train", "test"):
            raise ValueError(
                f"split must be 'train' or 'test', got '{split}'"
            )

        self.root_dir = root_dir
        self.split    = split

        # Choose the writer IDs that belong to this split.
        self.writers = TRAIN_WRITERS if split == "train" else TEST_WRITERS

        # ----------------------------------------------------------------
        # Absolute paths to the two image folders.
        # ----------------------------------------------------------------
        self.org_dir  = os.path.join(root_dir, "full_org")    # genuine images
        self.forg_dir = os.path.join(root_dir, "full_forg")   # forgery images

        # Sanity-check that the folders actually exist so the error message
        # is clear if someone points root_dir at the wrong location.
        for folder in (self.org_dir, self.forg_dir):
            if not os.path.isdir(folder):
                raise FileNotFoundError(
                    f"Expected image folder not found: {folder}\n"
                    f"Make sure root_dir='{root_dir}' is correct."
                )

        # ----------------------------------------------------------------
        # Build the list of pairs.  This is done once at construction time.
        # self.pairs is a Python list of  (path_A, path_B, label)  tuples.
        # ----------------------------------------------------------------
        self.pairs = self._build_pairs(seed)

    # ------------------------------------------------------------------
    # Private helper: enumerate all pairs for this split
    # ------------------------------------------------------------------

    def _build_pairs(self, seed: int):
        """
        Build and return the complete list of (path_A, path_B, label) tuples.

        Strategy per writer
        -------------------
        Positive pairs  (label = 1):
          All unique unordered pairs of genuine signatures.
          C(24, 2) = 24 × 23 / 2 = 276 pairs per writer.
          We use every possible positive pair — no sampling needed.

        Negative pairs  (label = 0):
          Each genuine signature can be paired with each of the 24 forgeries,
          giving 24 × 24 = 576 possible pairs per writer.
          We randomly sample 276 of them (= same as positive count) so the
          two classes stay balanced.

        The final list is shuffled so positive and negative pairs are mixed.
        """

        # Create a private RNG with our seed so we don't disturb any global
        # random state that other parts of the code might rely on.
        rng = random.Random(seed)

        all_pairs = []   # will hold every (path_A, path_B, label) triple

        for writer_id in self.writers:

            # ---- Step A: collect file paths for this writer ------------

            # Genuine images: original_{writer_id}_{1..24}.png
            genuine_paths = [
                os.path.join(
                    self.org_dir,
                    f"original_{writer_id}_{i}.png"
                )
                for i in range(1, SAMPLES_EACH + 1)   # i = 1, 2, …, 24
            ]

            # Forgery images: forgeries_{writer_id}_{1..24}.png
            forgery_paths = [
                os.path.join(
                    self.forg_dir,
                    f"forgeries_{writer_id}_{i}.png"
                )
                for i in range(1, SAMPLES_EACH + 1)
            ]

            # ---- Step B: positive pairs  (genuine vs genuine) ----------

            # itertools.combinations(sequence, 2) generates all pairs of
            # elements where order does NOT matter and elements are not
            # repeated.  For 24 items: (img1,img2), (img1,img3), … (img23,img24).
            positive_pairs = [
                (path_a, path_b, 1)                          # label 1 = genuine
                for path_a, path_b in itertools.combinations(genuine_paths, 2)
            ]
            # len(positive_pairs) == C(24, 2) == 276

            # ---- Step C: negative pairs  (genuine vs forgery) ----------

            # List every possible (genuine, forgery) combination for this writer.
            # Using a nested list comprehension: for each genuine image,
            # pair it with each forgery image.
            all_negative = [
                (path_g, path_f, 0)                          # label 0 = forgery
                for path_g in genuine_paths
                for path_f in forgery_paths
            ]
            # len(all_negative) == 24 × 24 == 576

            # Sample exactly as many negatives as there are positives.
            # min() is a safeguard — in practice 576 ≥ 276 always.
            n_sample       = min(len(positive_pairs), len(all_negative))
            negative_pairs = rng.sample(all_negative, n_sample)

            # ---- Step D: add both groups to the master list ------------
            all_pairs.extend(positive_pairs)
            all_pairs.extend(negative_pairs)

        # Shuffle the master list so both classes appear interleaved.
        # Without shuffling the first half of the list would be all positives
        # and the second half all negatives, which can confuse optimisers.
        rng.shuffle(all_pairs)

        return all_pairs

    # ------------------------------------------------------------------
    # PyTorch Dataset interface — the two methods PyTorch requires
    # ------------------------------------------------------------------

    def __len__(self):
        """
        Return the total number of pairs in this split.

        PyTorch's DataLoader calls this to know how many items the dataset
        contains so it can divide them into batches and shuffle correctly.
        """
        return len(self.pairs)

    def __getitem__(self, idx: int):
        """
        Load, preprocess, and return the pair at position `idx`.

        Parameters
        ----------
        idx : int
            Integer index chosen by the DataLoader (0 … len(self) - 1).

        Returns
        -------
        img1 : torch.Tensor  –  shape (1, 224, 224),  dtype float32,  range [0, 1]
        img2 : torch.Tensor  –  shape (1, 224, 224),  dtype float32,  range [0, 1]
        label : torch.Tensor –  scalar,  dtype int64,  value 0 or 1
        """

        # Retrieve the pre-built (path_A, path_B, label) triple.
        path_a, path_b, label = self.pairs[idx]

        # ----------------------------------------------------------------
        # Load and preprocess both images.
        #
        # preprocess_signature() reads the PNG from disk, converts it to
        # grayscale, binarises it with Otsu's method, crops to the ink
        # bounding box, pads to a square, and resizes to 224 × 224.
        #
        # Result: uint8 NumPy array,  shape (224, 224),
        #         0 = ink (dark),  255 = background (white).
        # ----------------------------------------------------------------
        array_a = preprocess_signature(path_a)   # shape (224, 224)
        array_b = preprocess_signature(path_b)

        # ----------------------------------------------------------------
        # Convert NumPy arrays to PyTorch tensors.
        #
        # Step 1 – Cast to float32 and scale from [0, 255] → [0.0, 1.0].
        #           Neural networks train much better on small floating-point
        #           values than on raw 0-255 integers.
        #
        # Step 2 – Add a channel dimension with unsqueeze(0).
        #           PyTorch convolution layers expect tensors in the form
        #           (C, H, W)  i.e.  (channels, height, width).
        #           Grayscale has 1 channel, so we go from (224, 224) →
        #           (1, 224, 224).
        # ----------------------------------------------------------------
        img1 = torch.from_numpy(array_a.astype(np.float32) / 255.0).unsqueeze(0)
        img2 = torch.from_numpy(array_b.astype(np.float32) / 255.0).unsqueeze(0)

        # Wrap the integer label in a tensor.  dtype=torch.long (int64) is
        # required by most PyTorch loss functions such as CrossEntropyLoss.
        label_tensor = torch.tensor(label, dtype=torch.long)

        return img1, img2, label_tensor


# ---------------------------------------------------------------------------
# Quick sanity-check — run this file directly: python dataset.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from torch.utils.data import DataLoader

    # The data folders are assumed to be next to this script.
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

    print("=" * 62)
    print("CEDAR Signature Dataset – sanity check")
    print("=" * 62)

    for split in ("train", "test"):
        print(f"\n--- {split.upper()} SPLIT ---")

        ds = CEDARSignatureDataset(root_dir=SCRIPT_DIR, split=split)

        # Count positives and negatives to confirm balance.
        n_pos = sum(1 for _, _, lbl in ds.pairs if lbl == 1)
        n_neg = sum(1 for _, _, lbl in ds.pairs if lbl == 0)

        writers = TRAIN_WRITERS if split == "train" else TEST_WRITERS
        print(f"  Writers           : {writers[0]} – {writers[-1]}")
        print(f"  Total pairs       : {len(ds)}")
        print(f"  Genuine-Genuine   : {n_pos}")
        print(f"  Genuine-Forgery   : {n_neg}")

        # Load one sample to confirm shapes and value ranges.
        img1, img2, lbl = ds[0]
        print(f"  img1 shape        : {tuple(img1.shape)}   (C, H, W)")
        print(f"  img1 dtype        : {img1.dtype}")
        print(f"  img1 value range  : [{img1.min():.3f}, {img1.max():.3f}]")
        print(f"  label             : {lbl.item()}  (0=forgery pair, 1=genuine pair)")

        # Wrap in a DataLoader and pull one mini-batch to test end-to-end.
        loader   = DataLoader(ds, batch_size=8, shuffle=True)
        b1, b2, bl = next(iter(loader))
        print(f"  Batch img shape   : {tuple(b1.shape)}   (N, C, H, W)")
        print(f"  Batch label shape : {tuple(bl.shape)}")

    print("\n" + "=" * 62)
    print("All checks passed.")
    print("=" * 62)
