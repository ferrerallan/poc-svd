# poc-svd

A POC of **SVD (Singular Value Decomposition)** applied to an image.

The idea: a grayscale image is just a matrix of numbers. SVD breaks that matrix
apart into a stack of **layers** ordered by importance. Adding up the first `N`
layers already gets you almost the whole image back — using far fewer numbers
than the original.

## What the script does

[svd_imagem.py](svd_imagem.py) reads `entrada.png`, converts it to grayscale
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
python svd_imagem.py
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

The three variables live at the top of [svd_imagem.py:25-27](svd_imagem.py#L25-L27):

| Variable     | Default         | What it is                                              |
| ------------ | --------------- | ------------------------------------------------------- |
| `IMAGE_PATH` | `"entrada.png"` | input image (any format Pillow can open)                |
| `NUM_LAYERS` | `35`            | how many layers to generate (capped at `min(height, width)`) |
| `OUTPUT_DIR` | `"saida"`       | folder where everything gets saved                      |

## Layout

```
.
├── svd_imagem.py       # the whole script
├── entrada.png         # input image
├── saida/
│   ├── layer_01.png    # layer 1 on its own (visualization)
│   ├── layer_01.npy    # layer 1 on its own (raw numbers)
│   ├── ...
│   └── reconstructed.png
└── README.md
```

## How it works

SVD decomposes the image matrix `W` into `W = U · Σ · Vᵀ`. Each layer `i` is the
outer product of a column of `U` with a row of `Vᵀ`, scaled by the weight `σᵢ`:

```python
layer_i = σᵢ * np.outer(U[:, i], Vt[i])
```

The `σ` weights come out sorted largest to smallest — which is why layer 1
carries the "skeleton" of the image and the later ones only add fine detail.

Summing the first `N` layers is the same as the truncated matrix multiplication:

```python
reconstructed = (U[:, :N] * σ[:N]) @ Vt[:N, :]
```

### About the layer PNGs

A single layer has **both negative and positive** values. To turn it into an
image, the script centers zero on mid gray (128) and rescales by the largest
absolute value ([`to_visible_image`](svd_imagem.py#L55-L62)). That's **for
visualization only** — the real numbers only survive in the `.npy` files.

## Report metrics

- **reconstruction** — each layer costs `height + width + 1` numbers (one column
  of `U`, one row of `Vᵀ` and one `σ`), against `height × width` for the full image
- **image captured** — the retained "energy", `Σσᵢ² (used) / Σσᵢ² (all)`
