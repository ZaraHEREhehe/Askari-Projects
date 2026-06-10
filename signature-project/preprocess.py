"""
preprocess.py
=============
Turns a raw signature scan into a clean, fixed-size image ready for a neural
network.  The pipeline is:

  raw image  →  grayscale  →  binary (Otsu)  →  crop  →  pad  →  resize 224×224

Run this file directly to test on 20 random images from data/full_org.
Results (side-by-side before/after) are saved to check/.
"""

import os
import random

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Core preprocessing function
# ---------------------------------------------------------------------------

def preprocess_signature(image_path: str) -> np.ndarray:
    """
    Load a signature image and return a clean 224×224 binary image.

    Parameters
    ----------
    image_path : str
        Path to the input image (any format OpenCV can read).

    Returns
    -------
    np.ndarray
        A uint8 array with shape (224, 224).
        Pixel value 0   = ink (dark).
        Pixel value 255 = background (white).
    """

    # ------------------------------------------------------------------
    # Step 1 – Load the image from disk.
    # cv2.imread returns a NumPy array in BGR colour order.
    # If the file doesn't exist or can't be decoded, it returns None.
    # ------------------------------------------------------------------
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"OpenCV could not open image: {image_path}")

    # ------------------------------------------------------------------
    # Step 2 – Convert to grayscale.
    # A colour image has three channels (Blue, Green, Red).  We collapse
    # them into a single channel where each pixel is just a brightness
    # value between 0 (black) and 255 (white).
    # ------------------------------------------------------------------
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # ------------------------------------------------------------------
    # Step 3 – Binarise with Otsu's threshold.
    #
    # "Binarise" means convert every pixel to either pure black (0) or
    # pure white (255) — no shades of grey.
    #
    # Otsu's method automatically picks the best threshold value by
    # analysing the histogram of the image.  We don't need to choose the
    # threshold manually.
    #
    # After thresholding:
    #   • pixels BELOW the threshold  → 0   (ink / foreground)
    #   • pixels ABOVE the threshold  → 255 (background)
    #
    # cv2.THRESH_BINARY_INV makes ink dark and background white, which
    # matches the convention used in most signature datasets.
    # ------------------------------------------------------------------
    _, binary = cv2.threshold(
        gray,
        0,                          # ignored when using Otsu
        255,                        # value assigned to "background" pixels
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    # After this step binary contains: 0 where ink is, 255 where background is.
    # But cv2.THRESH_BINARY sets pixels > threshold to 255.
    # For typical scans the background is bright, so the background gets 255
    # and the ink gets 0.  That is exactly what we want.
    # If your images are inverted (dark background) you would swap to
    # cv2.THRESH_BINARY_INV instead.

    # ------------------------------------------------------------------
    # Step 4 – Find the bounding box of the ink and crop to it.
    #
    # We look at where the ink pixels are (value == 0) and find the
    # smallest rectangle that contains all of them.
    # This removes empty white margins so the signature fills the frame.
    #
    # np.where returns the row/column indices of every ink pixel.
    # np.min/max of those indices give us the bounding box corners.
    # ------------------------------------------------------------------
    ink_rows, ink_cols = np.where(binary == 0)   # ink pixels are 0

    if ink_rows.size == 0:
        # Blank or all-white image — nothing to crop, return a white square.
        return np.full((224, 224), 255, dtype=np.uint8)

    row_min, row_max = int(ink_rows.min()), int(ink_rows.max())
    col_min, col_max = int(ink_cols.min()), int(ink_cols.max())

    # Slice the binary image to keep only the tight bounding box.
    cropped = binary[row_min : row_max + 1, col_min : col_max + 1]

    # ------------------------------------------------------------------
    # Step 5 – Pad to a square while preserving the aspect ratio.
    #
    # If we simply stretched the cropped rectangle to 224×224 we would
    # distort the signature (e.g. tall skinny writing would look fat).
    # Instead we add white padding on the shorter sides to make the image
    # square first, then resize.
    #
    # We also add a small fixed margin (PADDING px on each side) so the
    # ink doesn't touch the very edge of the canvas.
    # ------------------------------------------------------------------
    PADDING = 20   # extra white border in pixels before resizing

    h, w = cropped.shape

    # Determine the size of the square canvas: longest side + 2×padding.
    canvas_size = max(h, w) + 2 * PADDING

    # Create an all-white canvas of that size.
    canvas = np.full((canvas_size, canvas_size), 255, dtype=np.uint8)

    # Calculate where to paste the cropped signature so it sits centred.
    top  = (canvas_size - h) // 2
    left = (canvas_size - w) // 2

    canvas[top : top + h, left : left + w] = cropped

    # ------------------------------------------------------------------
    # Step 6 – Resize to the target size (224×224).
    #
    # cv2.INTER_AREA is the best interpolation method when shrinking an
    # image; it avoids aliasing (jagged edges) by averaging pixels.
    # ------------------------------------------------------------------
    TARGET = 224
    final = cv2.resize(canvas, (TARGET, TARGET), interpolation=cv2.INTER_AREA)

    return final


# ---------------------------------------------------------------------------
# Helper: build a side-by-side comparison image for visual inspection
# ---------------------------------------------------------------------------

def make_comparison(original_bgr: np.ndarray, processed: np.ndarray) -> np.ndarray:
    """
    Stack the original (colour) and processed (grayscale) images side by side.
    Both are resized to the same height for a clean comparison.

    Returns a BGR image suitable for cv2.imwrite.
    """
    TARGET_H = 224

    # Resize original to the same height, keeping its aspect ratio.
    orig_h, orig_w = original_bgr.shape[:2]
    scale = TARGET_H / orig_h
    orig_resized = cv2.resize(
        original_bgr,
        (int(orig_w * scale), TARGET_H),
        interpolation=cv2.INTER_AREA,
    )

    # Convert the single-channel processed image to BGR so we can stack it
    # next to the colour original.
    processed_bgr = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)

    # Add a thin vertical divider line between the two panels.
    divider = np.full((TARGET_H, 4, 3), 200, dtype=np.uint8)   # light-grey line

    comparison = np.hstack([orig_resized, divider, processed_bgr])
    return comparison


# ---------------------------------------------------------------------------
# Test section – runs when you execute this file directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    # Paths relative to this script's location.
    SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR    = os.path.join(SCRIPT_DIR, "data", "full_org")
    CHECK_DIR   = os.path.join(SCRIPT_DIR, "check")

    # Fall back to full_org at the project root if data/full_org doesn't exist.
    if not os.path.isdir(DATA_DIR):
        DATA_DIR = os.path.join(SCRIPT_DIR, "full_org")

    if not os.path.isdir(DATA_DIR):
        raise FileNotFoundError(
            f"Could not find image folder.  Expected: {DATA_DIR}"
        )

    # Create the output folder if it doesn't exist yet.
    os.makedirs(CHECK_DIR, exist_ok=True)

    # Collect all image files in the data folder.
    VALID_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    all_images = [
        f for f in os.listdir(DATA_DIR)
        if os.path.splitext(f)[1].lower() in VALID_EXTENSIONS
    ]

    if len(all_images) == 0:
        raise FileNotFoundError(f"No images found in {DATA_DIR}")

    # Pick 20 at random (or fewer if the folder has less than 20 images).
    sample_size = min(20, len(all_images))
    selected    = random.sample(all_images, sample_size)

    print(f"Found {len(all_images)} images in {DATA_DIR}")
    print(f"Processing {sample_size} random images …\n")

    for i, filename in enumerate(selected, start=1):
        image_path = os.path.join(DATA_DIR, filename)

        # Load the original in colour for the "before" panel.
        original_bgr = cv2.imread(image_path)

        # Run the preprocessing pipeline.
        processed = preprocess_signature(image_path)

        # Build the before/after comparison and save it.
        comparison = make_comparison(original_bgr, processed)
        stem       = os.path.splitext(filename)[0]   # filename without extension
        out_path   = os.path.join(CHECK_DIR, f"{stem}_check.png")
        cv2.imwrite(out_path, comparison)

        print(f"[{i:2d}/{sample_size}]  {filename}  ->  {os.path.basename(out_path)}")

    print(f"\nDone.  Comparisons saved to: {CHECK_DIR}")
