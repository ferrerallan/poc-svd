"""
* what? - Warm-up (experiment 0): SVD of an image, saved layer by layer and as a reconstruction from the first N layers.
* why?  - A grayscale image is a matrix; SVD splits it into layers ordered by importance, the idea the whole thesis builds on.
* how?  - np.linalg.svd gives U, sigma and Vt; each layer is sigma_i * outer(u_i, v_i); the sum of N layers rebuilds the image.

For the first NUM_LAYERS layers the script:
  1) saves the image of EACH layer on its own (layer_01.png, layer_02.png, ...) and its raw numbers (layer_01.npy, ...)
  2) saves the reconstruction using the N layers added together (reconstructed.png)
The larger NUM_LAYERS is, the closer the reconstruction gets to the original.
Dependencies: numpy and Pillow (pip install numpy pillow).
Run from anywhere:  python3 doutorado/experimento0/svd_image.py
"""
import os
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent

# ------------------------------------------------------------------
# VARIABLES YOU FEED IN
# ------------------------------------------------------------------
IMAGE_PATH = HERE / "entrada.png"   # the input image (the file name is a frozen data name)
NUM_LAYERS = 35                     # how many layers to generate (1, 2, ... n)
OUTPUT_DIR = HERE / "saida"         # folder where everything gets saved
# ------------------------------------------------------------------


def load_image_as_matrix(path):
    """
    * what? - Reads an image, converts it to grayscale and returns its pixel matrix.
    * why?  - SVD works on a matrix of numbers; each number here is the brightness of a pixel, from 0 to 255.
    * how?  - Pillow opens and converts to mode L; numpy turns it into a float array.
    """
    #region 1 - Load and convert
    # what? - Returns the float matrix.
    # how?  - dtype=float so the later SVD runs in floating point.
    gray_image = Image.open(path).convert("L")
    return np.asarray(gray_image, dtype=float)
    #endregion


def build_layer(matrix_U, weights, matrix_Vt, layer_number):
    """
    * what? - Builds one whole layer, already weighted: weight * (column x row).
    * why?  - Each layer is one ingredient triple of the SVD: a column of U, a row of Vt and a weight sigma.
    * how?  - layer_number counts the human way (1, 2, 3...), so the real index is layer_number - 1; np.outer builds the layer.
    """
    #region 1 - Pick the three ingredients
    # what? - Column u, row v and weight sigma of this layer.
    # how?  - Python indexes from zero, hence index = layer_number - 1.
    index = layer_number - 1
    column = matrix_U[:, index]    # the u vector of this layer
    row = matrix_Vt[index]         # the v vector of this layer
    weight = weights[index]        # the sigma of this layer
    #endregion

    #region 2 - Combine them
    # what? - Returns the weighted outer product.
    # how?  - The layer has the same shape as the image.
    return weight * np.outer(column, row)
    #endregion


def to_visible_image(layer):
    """
    * what? - Converts the raw numbers of a layer into an 8-bit image you can look at.
    * why?  - A layer has negative and positive values; the real numbers are lost in this conversion, so it is for display only.
    * how?  - Centers zero on mid gray (128) and rescales by the largest absolute value, then clips to 0-255.
    """
    #region 1 - Rescale to a visible range
    # what? - Returns a uint8 matrix.
    # how?  - "or 1" avoids a division by zero for an all-zero layer.
    largest_absolute_value = np.abs(layer).max() or 1
    visible = 128 + 127 * (layer / largest_absolute_value)
    return np.clip(visible, 0, 255).astype(np.uint8)
    #endregion


def save_png(uint8_matrix, path):
    """
    * what? - Saves an 8-bit matrix as a PNG file.
    * why?  - Layers and the reconstruction are saved as images.
    * how?  - Image.fromarray(...).save(path).
    """
    #region 1 - Save the file
    # what? - Writes the PNG.
    # how?  - Pillow infers the format from the file extension.
    Image.fromarray(uint8_matrix).save(path)
    #endregion


def main():
    """
    * what? - Runs the whole warm-up: loads the image, decomposes it, saves the layers and the reconstruction, prints a report.
    * why?  - Shows how few layers are enough to recover most of an image, the intuition behind studying weights with SVD.
    * how?  - SVD once; one file pair per layer; a truncated matrix product for the reconstruction; energy kept as sum of sigma squared.
    """
    #region 1 - Load the image and size the work
    # what? - Matrix, its shape, the number of layers to build and the output folder.
    # how?  - The layer count is capped at min(height, width), the maximum rank.
    image_matrix = load_image_as_matrix(IMAGE_PATH)
    height, width = image_matrix.shape

    total_possible_layers = min(height, width)
    layer_count = max(1, min(NUM_LAYERS, total_possible_layers))

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    #endregion

    #region 2 - Decompose with SVD
    # what? - Two matrices and one list of weights.
    # how?  - In theory W = U . Sigma . Vt: matrix_U gives the COLUMN of each layer, weights the sigmas (largest to smallest), matrix_Vt the ROW.
    matrix_U, weights, matrix_Vt = np.linalg.svd(image_matrix, full_matrices=False)

    # width of the number in the file name (layer_01, layer_02, ...)
    name_digits = len(str(layer_count))
    #endregion

    #region 3 - Save each layer on its own
    # what? - From layer 1 up to layer_count, in two formats.
    # how?  - .npy keeps the raw numbers (weight baked in) to add up later; .png is only to see the pattern.
    for layer_number in range(1, layer_count + 1):
        layer = build_layer(matrix_U, weights, matrix_Vt, layer_number)
        suffix = f"{layer_number:0{name_digits}d}"
        np.save(os.path.join(OUTPUT_DIR, f"layer_{suffix}.npy"), layer)
        save_png(to_visible_image(layer), os.path.join(OUTPUT_DIR, f"layer_{suffix}.png"))
    #endregion

    #region 4 - Reconstruct with the first layer_count layers
    # what? - The sum of the first layers, saved as reconstructed.png.
    # how?  - (used_matrix_U * used_weights) puts the sigma into each column; the @ matrix product builds and adds all layers at once.
    used_matrix_U = matrix_U[:, :layer_count]
    used_weights = weights[:layer_count]
    used_matrix_Vt = matrix_Vt[:layer_count, :]

    reconstructed_image = (used_matrix_U * used_weights) @ used_matrix_Vt
    reconstructed_image = np.clip(reconstructed_image, 0, 255).astype(np.uint8)
    save_png(reconstructed_image, os.path.join(OUTPUT_DIR, "reconstructed.png"))
    #endregion

    #region 5 - Print the report
    # what? - Compression and captured-energy summary.
    # how?  - Each layer costs height + width + 1 numbers (one column, one row and one sigma).
    numbers_in_original = height * width
    numbers_stored = layer_count * (height + width + 1)
    stored_ratio = 100 * numbers_stored / numbers_in_original
    captured_energy = 100 * np.sum(used_weights ** 2) / np.sum(weights ** 2)

    print(f"image:             {width} x {height}  ({numbers_in_original:,} numbers)")
    print(f"layers generated:  {layer_count} of {total_possible_layers}")
    print(f"reconstruction:    {numbers_stored:,} numbers ({stored_ratio:.1f}% of the original)")
    print(f"image captured:    {captured_energy:.1f}%")
    print(f"everything saved:  {Path(OUTPUT_DIR).relative_to(HERE)}/")
    #endregion


if __name__ == "__main__":
    main()
