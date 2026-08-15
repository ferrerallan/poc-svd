"""
POC: SVD on an image.

The idea: a grayscale image is just a matrix of numbers.
SVD breaks it apart into a stack of layers ordered by importance.

This script does two things for the first NUM_LAYERS layers:
  1) saves the image of EACH layer on its own (layer_01.png, layer_02.png, ...)
     and also the raw numbers of each layer (layer_01.npy, ...)
  2) saves the reconstruction using the N layers added together (reconstructed.png)

The larger NUM_LAYERS is, the closer the reconstruction gets to the original.

Dependencies: numpy and Pillow
    pip install numpy pillow
"""
 
import os
import numpy as np
from PIL import Image

# ------------------------------------------------------------------
# VARIABLES YOU FEED IN
# ------------------------------------------------------------------
IMAGE_PATH  = "entrada.png"   # the input image
NUM_LAYERS  = 35              # how many layers to generate (1, 2, ... n)
OUTPUT_DIR  = "saida"         # folder where everything gets saved
# ------------------------------------------------------------------


def load_image_as_matrix(path):
    """Reads the image, converts it to grayscale and returns the pixel matrix
    (each number is the brightness of a pixel, from 0 to 255)."""
    gray_image = Image.open(path).convert("L")
    return np.asarray(gray_image, dtype=float)


def build_layer(matrix_U, weights, matrix_Vt, layer_number):
    """Builds a whole layer, already weighted: weight * (column x row).

    layer_number counts the human way (1, 2, 3...). Python indexes from zero,
    so the real position is layer_number - 1.

    From each piece the SVD takes one ingredient of this layer:
      - a COLUMN of matrix_U
      - a ROW of matrix_Vt
      - a WEIGHT (number) from the weights list"""
    index  = layer_number - 1
    column = matrix_U[:, index]    # the u vector of this layer
    row    = matrix_Vt[index]      # the v vector of this layer
    weight = weights[index]        # the sigma of this layer
    return weight * np.outer(column, row)


def to_visible_image(layer):
    """Converts the raw numbers of a layer into something you can actually look at.
    A layer has negative and positive values, so we center zero on mid gray (128)
    and rescale by the largest absolute value. This is ONLY for visualization:
    the real numbers are lost in this conversion."""
    largest_absolute_value = np.abs(layer).max() or 1
    visible = 128 + 127 * (layer / largest_absolute_value)
    return np.clip(visible, 0, 255).astype(np.uint8)


def save_png(uint8_matrix, path):
    Image.fromarray(uint8_matrix).save(path)


# ------------------------------------------------------------------
# 1) load the image
image_matrix = load_image_as_matrix(IMAGE_PATH)
height, width = image_matrix.shape

total_possible_layers = min(height, width)
layer_count = max(1, min(NUM_LAYERS, total_possible_layers))

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 2) the SVD returns TWO matrices and ONE list of weights.
#    In theory: W = U . Sigma . Vt
#      matrix_U  -> the U matrix   (where the COLUMN of each layer comes from)
#      weights   -> the sigmas     (LIST of numbers, largest to smallest)
#      matrix_Vt -> the Vt matrix  (where the ROW of each layer comes from)
matrix_U, weights, matrix_Vt = np.linalg.svd(image_matrix, full_matrices=False)

# width of the number in the file name (layer_01, layer_02, ...)
name_digits = len(str(layer_count))

# 3) saves each layer on its own, from 1 up to layer_count, in two formats:
#      .npy -> raw numbers (with the weight baked in), useful to add up later
#      .png -> version just to see the pattern
for layer_number in range(1, layer_count + 1):
    layer = build_layer(matrix_U, weights, matrix_Vt, layer_number)
    suffix = f"{layer_number:0{name_digits}d}"
    np.save(os.path.join(OUTPUT_DIR, f"layer_{suffix}.npy"), layer)
    save_png(to_visible_image(layer), os.path.join(OUTPUT_DIR, f"layer_{suffix}.png"))

# 4) reconstruction with the first layer_count layers added together.
#    (used_matrix_U * used_weights) puts the sigma into each column;
#    the @ (matrix multiplication) builds each layer and adds them all at once.
used_matrix_U  = matrix_U[:, :layer_count]
used_weights   = weights[:layer_count]
used_matrix_Vt = matrix_Vt[:layer_count, :]

reconstructed_image = (used_matrix_U * used_weights) @ used_matrix_Vt
reconstructed_image = np.clip(reconstructed_image, 0, 255).astype(np.uint8)
save_png(reconstructed_image, os.path.join(OUTPUT_DIR, "reconstructed.png"))

# 5) report
numbers_in_original = height * width
numbers_stored      = layer_count * (height + width + 1)
stored_ratio        = 100 * numbers_stored / numbers_in_original
captured_energy     = 100 * np.sum(used_weights ** 2) / np.sum(weights ** 2)

print(f"image:             {width} x {height}  ({numbers_in_original:,} numbers)")
print(f"layers generated:  {layer_count} of {total_possible_layers}")
print(f"reconstruction:    {numbers_stored:,} numbers ({stored_ratio:.1f}% of the original)")
print(f"image captured:    {captured_energy:.1f}%")
print(f"everything saved:  {OUTPUT_DIR}/")
