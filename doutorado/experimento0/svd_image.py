"""
* what? - Warm-up (experiment 0): SVD of an image, saved layer by layer and as a reconstruction from the first N layers.
* why?  - A grayscale image is a matrix; SVD splits it into layers ordered by importance, the idea the whole thesis builds on.
* how?  - np.linalg.svd gives U, the sigmas and Vt; layer i is sigma_i * outer(u_i, v_i); the sum of N layers rebuilds the image.

What the SVD returns for a matrix with `height` rows and `width` columns (here 296 rows x 336 columns):
  U      - one COLUMN per layer, each with `height` numbers: the layer's pattern down the rows of the image
  sigmas - one number per layer, sorted largest to smallest: how much each layer weighs
  Vt     - one ROW per layer, each with `width` numbers: the layer's pattern across the columns of the image
Layer i is sigmas[i] * outer(U[:, i], Vt[i]); pixel (r, c) of that layer is sigmas[i] * U[r, i] * Vt[i, c].
Adding every layer gives the matrix back exactly; adding only the first N gives the best N-layer approximation.

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
    * how?  - Pillow opens and converts to mode L; numpy turns it into a float array, one row per pixel row.
    """
    #region 1 - Load and convert
    # what? - Returns the float matrix, shaped (height, width).
    # how?  - dtype=float so the later SVD runs in floating point.
    gray_image = Image.open(path).convert("L")
    return np.asarray(gray_image, dtype=float)
    #endregion


def decompose_into_layers(matrix):
    """
    * what? - Runs the SVD on the matrix and returns its three parts: U, the sigmas and Vt.
    * why?  - Every layer is built from these three parts, so this is the place that explains the SVD call.
    * how?  - np.linalg.svd(matrix, full_matrices=False) returns U (height x k), k sigmas and Vt (k x width).
    """
    #region 1 - Run the SVD
    # what? - The three arrays such that matrix == U @ np.diag(sigmas) @ Vt.
    # how?  - np.linalg.svd takes the matrix as its only data argument; the result is exact, no approximation.
    # matrix_U:  one column per layer; column i has one number per image row (the pattern down the height)
    # sigmas:    one number per layer, largest first; sigmas[i] is how much layer i weighs
    # matrix_Vt: one row per layer; row i has one number per image column (the pattern across the width)
    # k = min(height, width) is how many layers can weigh anything; full_matrices=False keeps only those,
    # so U is height x k and Vt is k x width; full_matrices=True would pad both to square with layerless rows
    matrix_U, sigmas, matrix_Vt = np.linalg.svd(matrix, full_matrices=False)
    return matrix_U, sigmas, matrix_Vt
    #endregion


def build_layer(matrix_U, sigmas, matrix_Vt, layer_number):
    """
    * what? - Builds one whole layer, already weighted: sigma_i times (column i of U) x (row i of Vt).
    * why?  - Each layer is one triple of the SVD: a column of U, a row of Vt and a sigma. Here they meet.
    * how?  - Human numbering (1, 2, 3...) becomes index layer_number - 1; np.outer builds the weighted layer.
    """
    #region 1 - Pick the three ingredients
    # what? - Column i of U, row i of Vt and the weight sigma_i of this layer.
    # how?  - Python indexes from zero, hence index = layer_number - 1.
    index = layer_number - 1
    u_column = matrix_U[:, index]    # one number per image row: the layer's pattern down the height
    v_row = matrix_Vt[index]         # one number per image column: the layer's pattern across the width
    sigma = sigmas[index]            # how much this layer weighs
    #endregion

    #region 2 - Combine them
    # what? - Returns the weighted outer product, a matrix with the same shape as the image.
    # how?  - np.outer(u_column, v_row)[r, c] is u_column[r] * v_row[c]; the layer is that times sigma.
    return sigma * np.outer(u_column, v_row)
    #endregion


def to_visible_image(layer):
    """
    * what? - Converts the raw numbers of a layer into an 8-bit image you can look at.
    * why?  - A layer mixes negative and positive values; the conversion loses them, so it is display only.
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
    * what? - Runs the warm-up end to end: load, SVD, save every layer and the reconstruction, print a report.
    * why?  - Shows how few layers recover most of an image, the intuition behind studying weights with SVD.
    * how?  - One SVD; a .npy/.png pair per layer; a truncated product rebuilds; energy is sum(sigma^2).
    """
    #region 1 - Load the image and size the work
    # what? - Matrix, its shape, the number of layers to build and the output folder.
    # how?  - The layer count is capped at min(height, width), the most layers that can weigh anything.
    image_matrix = load_image_as_matrix(IMAGE_PATH)
    height, width = image_matrix.shape

    total_possible_layers = min(height, width)
    layer_count = max(1, min(NUM_LAYERS, total_possible_layers))

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    #endregion

    #region 2 - Decompose with SVD
    # what? - U, the sigmas and Vt of the image matrix.
    # how?  - decompose_into_layers explains each part; the sigmas come out sorted largest first.
    matrix_U, sigmas, matrix_Vt = decompose_into_layers(image_matrix)
    #endregion

    #region 3 - Size the file-name suffix
    # what? - Number of digits for the layer number in file names (layer_01, layer_02, ...).
    # how?  - Width of layer_count as a string, so the files sort in layer order inside the folder.
    name_digits = len(str(layer_count))
    #endregion

    #region 4 - Save each layer on its own
    # what? - From layer 1 up to layer_count, in two formats.
    # how?  - .npy keeps the raw numbers (sigma baked in) to add up later; .png is only to see the pattern.
    for layer_number in range(1, layer_count + 1):
        layer = build_layer(matrix_U, sigmas, matrix_Vt, layer_number)
        suffix = f"{layer_number:0{name_digits}d}"
        np.save(os.path.join(OUTPUT_DIR, f"layer_{suffix}.npy"), layer)
        save_png(to_visible_image(layer), os.path.join(OUTPUT_DIR, f"layer_{suffix}.png"))
    #endregion

    #region 5 - Reconstruct with the first layer_count layers
    # what? - The sum of the first layers, saved as reconstructed.png.
    # how?  - (used_matrix_U * used_sigmas) bakes each sigma into its column; @ adds all layers at once.
    used_matrix_U = matrix_U[:, :layer_count]
    used_sigmas = sigmas[:layer_count]
    used_matrix_Vt = matrix_Vt[:layer_count, :]

    reconstructed_image = (used_matrix_U * used_sigmas) @ used_matrix_Vt
    reconstructed_image = np.clip(reconstructed_image, 0, 255).astype(np.uint8)
    save_png(reconstructed_image, os.path.join(OUTPUT_DIR, "reconstructed.png"))
    #endregion

    #region 6 - Print the report
    # what? - Compression and captured-energy summary.
    # how?  - Each layer costs height + width + 1 numbers (one column of U, one row of Vt and one sigma).
    numbers_in_original = height * width
    numbers_stored = layer_count * (height + width + 1)
    stored_ratio = 100 * numbers_stored / numbers_in_original
    captured_energy = 100 * np.sum(used_sigmas ** 2) / np.sum(sigmas ** 2)

    print(f"image:             {width} x {height}  ({numbers_in_original:,} numbers)")
    print(f"layers generated:  {layer_count} of {total_possible_layers}")
    print(f"reconstruction:    {numbers_stored:,} numbers ({stored_ratio:.1f}% of the original)")
    print(f"image captured:    {captured_energy:.1f}%")
    print(f"everything saved:  {Path(OUTPUT_DIR).relative_to(HERE)}/")
    #endregion


if __name__ == "__main__":
    main()
