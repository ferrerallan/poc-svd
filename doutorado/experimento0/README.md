# Experiment 0: SVD of an image (the warm-up)

This is the first stage of the research story: the warm-up that gave the repository its name (poc-svd). The full story is in [`artigo/index.html`](../../artigo/index.html).

A POC of **SVD (Singular Value Decomposition)** applied to an image.

The idea: a grayscale image is just a matrix of numbers. SVD breaks that matrix
apart into a stack of **layers** ordered by importance. Adding up the first `N`
layers already gets you almost the whole image back — using far fewer numbers
than the original.

## What the script does

[svd_image.py](svd_image.py) reads `entrada.png`, converts it to grayscale
and, for the first `NUM_LAYERS` layers:

1. saves **each layer on its own** in two formats:
   - `layer_NN.png` — a visible version, just to see the pattern
   - `layer_NN.npy` — the raw numbers (with the weight/sigma baked in), useful to add up later
2. saves the **reconstruction** of the `N` layers summed together as `reconstructed.png`
3. prints a compression report to the terminal

The larger `NUM_LAYERS` is, the closer the reconstruction gets to the original.

## Requirements

```bash
pip install numpy pillow
```

## Running it

```bash
python3 doutorado/experimento0/svd_image.py
```

Shape of the terminal output (with the 336 × 296 `entrada.png` and 35 layers):

```
image:             336 x 296  (99,456 numbers)
layers generated:  35 of 296
reconstruction:    22,155 numbers (22.3% of the original)
image captured:    <depends on the image>%
everything saved:  saida/
```

In other words: at this setting the reconstruction keeps only **22.3% of the
numbers** in the original image.

## Configuration

The three variables live at the top of [svd_image.py:31-33](svd_image.py#L31-L33):

| Variable     | Default         | What it is                                              |
| ------------ | --------------- | ------------------------------------------------------- |
| `IMAGE_PATH` | `entrada.png` (next to the script) | input image (any format Pillow can open)                |
| `NUM_LAYERS` | `35`            | how many layers to generate (capped at `min(height, width)`) |
| `OUTPUT_DIR` | `saida` (next to the script) | folder where everything gets saved                      |

## Layout

```
.
├── svd_image.py        # the whole script
├── entrada.png         # input image
├── saida/
│   ├── layer_01.png    # layer 1 on its own (visualization)
│   ├── layer_01.npy    # layer 1 on its own (raw numbers)
│   ├── ...
│   └── reconstructed.png
└── README.md
```

## How it works

The whole decomposition is one NumPy call, wrapped in
[`decompose_into_layers`](svd_image.py#L51-L67):

```python
matrix_U, sigmas, matrix_Vt = np.linalg.svd(matrix, full_matrices=False)
```

`np.linalg.svd` takes the image matrix (here 296 rows × 336 columns) and returns
three arrays such that `matrix == U · diag(σ) · Vᵀ`:

| Returned     | Shape     | What it is                                                                                        |
| ------------ | --------- | ------------------------------------------------------------------------------------------------- |
| `matrix_U`   | 296 × 296 | the matrix `U`: one **column** per layer; column `i` has one number per image row, the layer's pattern down the height |
| `sigmas`     | 296       | the `σ`: one number per layer, sorted largest to smallest; `σᵢ` is how much layer `i` weighs      |
| `matrix_Vt`  | 296 × 336 | the matrix `Vᵀ` (`V` transposed, as NumPy returns it): one **row** per layer; row `i` has one number per image column, the layer's pattern across the width |

`full_matrices=False` keeps only the `k = min(height, width) = 296` layers that
can carry weight. Without it, `Vᵀ` would come back square (336 × 336), padded
with 40 rows that belong to no layer and have no `σ`.

Each layer `i` is the outer product of column `i` of `U` with row `i` of `Vᵀ`,
scaled by the weight `σᵢ` ([`build_layer`](svd_image.py#L70-L89)):

```python
layer_i = sigmas[i] * np.outer(matrix_U[:, i], matrix_Vt[i])   # pixel (r, c) = sigma_i * U[r, i] * Vt[i, c]
```

The `σ` weights come out sorted largest to smallest — which is why layer 1
carries the "skeleton" of the image and the later ones only add fine detail.
Adding all 296 layers gives the exact image back; summing only the first `N`
is the same as the truncated matrix multiplication:

```python
reconstructed = (matrix_U[:, :N] * sigmas[:N]) @ matrix_Vt[:N, :]
```

### About the layer PNGs

A single layer has **both negative and positive** values. To turn it into an
image, the script centers zero on mid gray (128) and rescales by the largest
absolute value ([`to_visible_image`](svd_image.py#L92-L104)). That's **for
visualization only** — the real numbers only survive in the `.npy` files.

## Report metrics

- **reconstruction** — each layer costs `height + width + 1` numbers (one column
  of `U`, one row of `Vᵀ` and one `σ`), against `height × width` for the full image
- **image captured** — the retained "energy", `Σσᵢ² (used) / Σσᵢ² (all)`
