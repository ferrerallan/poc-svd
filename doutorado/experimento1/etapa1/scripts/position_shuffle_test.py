"""
* what? - Study test of session 4: does shuffling the 14 x 14 patches of a photo change the DINOv2 output, with and without position?
* why?  - Backs the claim that the position vector is what tells the model where each patch was; without it, order cannot matter.
* how?  - Rebuilds the photo with patches in a random order, compares the CLS token of original and shuffled photo by cosine.

The model is not modified: the "without position" variant is a deep copy whose position table is set to zero.
Seed 0 repeats the original test of 2026-10-05 (0.4799 with position, 1.0000 without); seeds 0 to 9 show how much 0.48 depends on the order.
Run from the repository root:
    python3 doutorado/experimento1/etapa1/scripts/position_shuffle_test.py | tee doutorado/experimento1/etapa1/estudos/posicao_embaralhada.log
Output: doutorado/experimento1/etapa1/estudos/posicao_embaralhada.json
"""
import copy
import json
import statistics as st
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoImageProcessor, Dinov2Model

ROOT = Path(__file__).resolve().parents[4]
PHOTO = ROOT / "doutorado/experimento0/entrada.png"
MODEL = ROOT / "models/dinov2-small"
OUTPUT = Path(__file__).resolve().parents[1] / "estudos" / "posicao_embaralhada.json"
SEEDS = list(range(10))


def shuffle_patches(x, seed):
    """
    * what? - Cuts a 224 x 224 photo tensor into 16 x 16 patches of 14 x 14, shuffles their order and puts the photo back together.
    * why?  - Gives the same photo content in a different patch order, which is what the position test needs.
    * how?  - unfold into patches, torch.randperm(256) under a fixed seed, then the reverse reshape.
    """
    #region 1 - Cut the photo into patches
    # what? - Tensor of shape (1, 256, 3, 14, 14), one entry per patch.
    # how?  - unfold on height and width, then permute so the patch index comes first.
    p = x.unfold(2, 14, 14).unfold(3, 14, 14)
    p = p.permute(0, 2, 3, 1, 4, 5).reshape(1, 256, 3, 14, 14)
    #endregion

    #region 2 - Shuffle and reassemble
    # what? - The photo with the patches in a random order.
    # how?  - The seed makes the order reproducible; the reshape undoes the cut.
    torch.manual_seed(seed)
    p = p[:, torch.randperm(256)]
    return p.reshape(1, 16, 16, 3, 14, 14).permute(0, 3, 1, 4, 2, 5).reshape(1, 3, 224, 224)
    #endregion


def cls_output(model, image):
    """
    * what? - Returns the final CLS token (384 numbers) the model produces for a photo.
    * why?  - The CLS token is the summary of the photo, the value compared in the test.
    * how?  - Forward pass without gradients; token 0 of the last hidden state.
    """
    #region 1 - Run the model
    # what? - The CLS vector.
    # how?  - no_grad keeps the test cheap and leaves the model untouched.
    with torch.no_grad():
        return model(pixel_values=image).last_hidden_state[0, 0]
    #endregion


def cosine(a, b):
    """
    * what? - Cosine similarity between two vectors, as a float (1 means the same direction).
    * why?  - It is the measure of "same output" used by the test.
    * how?  - torch.nn.functional.cosine_similarity along dimension 0.
    """
    #region 1 - Compute the cosine
    # what? - A plain Python float.
    # how?  - .item() converts the tensor.
    return torch.nn.functional.cosine_similarity(a, b, dim=0).item()
    #endregion


def main():
    """
    * what? - Runs the shuffle test for 10 random orders, prints a table and writes the JSON.
    * why?  - Turns the numbers quoted in session 4 into a reproducible result with its spread.
    * how?  - Loads the model with and without the position table, shuffles the patches per seed and compares the CLS tokens.
    """
    #region 1 - Load the photo and the two models
    # what? - The preprocessed photo, the model with position and a copy with the position table zeroed.
    # how?  - AutoImageProcessor resizes and normalizes the photo exactly as the training code does for tests.
    processor = AutoImageProcessor.from_pretrained(str(MODEL))
    with_position = Dinov2Model.from_pretrained(str(MODEL)).eval()
    without_position = copy.deepcopy(with_position)
    without_position.embeddings.position_embeddings.data.zero_()
    x = processor(images=Image.open(PHOTO).convert("RGB"), return_tensors="pt")["pixel_values"]
    #endregion

    #region 2 - Compare original and shuffled photo for every seed
    # what? - One cosine per model and seed.
    # how?  - The original photo output is computed once per model.
    base_with, base_without = cls_output(with_position, x), cls_output(without_position, x)
    cos_with, cos_without = [], []
    print("CLS output: original photo x photo with shuffled patches (cosine similarity)")
    print(f"{'seed':>4}  {'with position':>14}  {'without position':>17}")
    for seed in SEEDS:
        shuffled = shuffle_patches(x, seed)
        cos_with.append(cosine(base_with, cls_output(with_position, shuffled)))
        cos_without.append(cosine(base_without, cls_output(without_position, shuffled)))
        print(f"{seed:>4}  {cos_with[-1]:>14.4f}  {cos_without[-1]:>17.4f}")
    #endregion

    #region 3 - Summarize and write the result
    # what? - Mean, minimum and maximum over the seeds, saved with the per-seed values.
    # how?  - Seed 0 is the original test of session 4.
    summary = {"with_position": {"mean": st.mean(cos_with), "min": min(cos_with), "max": max(cos_with)},
               "without_position": {"mean": st.mean(cos_without), "min": min(cos_without), "max": max(cos_without)}}
    print(f"\nwith position:    mean {summary['with_position']['mean']:.4f}  min {summary['with_position']['min']:.4f}  max {summary['with_position']['max']:.4f}")
    print(f"without position: mean {summary['without_position']['mean']:.4f}  min {summary['without_position']['min']:.4f}  max {summary['without_position']['max']:.4f}")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps({"photo": str(PHOTO.relative_to(ROOT)), "model": str(MODEL.relative_to(ROOT)), "seeds": SEEDS,
                                  "cosine_with_position": cos_with, "cosine_without_position": cos_without, "summary": summary}, indent=2))
    print("written:", OUTPUT.relative_to(ROOT))
    #endregion


if __name__ == "__main__":
    main()
