"""
* what? - Measures how much the change of the WEIGHTS and the change of the DIRECTIONS repeat from one training to another.
* why?  - If trainings agree, the change is task signal; if not, it is noise from the seed.
* how?  - Per matrix and seed builds a weight-change vector and a direction-displacement vector, then averages pairwise cosines.

For every matrix and seed:
  weights:    vector of the 10 relative changes of the top singular values  (sigma_after - sigma_before) / sigma_before
  directions: displacement of the 10 main directions (U_after sign-aligned to U_before, minus U_before)
Then the cosine between the changes of each pair of seeds (45 pairs): 0 = no agreement, 1 = identical change.
Run from the repository root:  python3 doutorado/experimento1/common/reproducibility_weights_directions.py --stage 1|2
"""
import argparse
import itertools
import json
import statistics as st

import stages
import torch
from safetensors.torch import load_file

TOP = 10
SEEDS = list(range(10))


def kind(name):
    """
    * what? - Returns the matrix type of a tensor name, such as 'mlp.fc1' or 'query'.
    * why?  - Results are grouped by matrix type.
    * how?  - Joins the name parts between the block number and 'weight' and strips the attention prefixes.
    """
    #region 1 - Extract the type
    # what? - Cuts the checkpoint name down to the type.
    # how?  - Same rule as nickname() in compare.py.
    p = name.split(".")
    return ".".join(p[3:-1]).replace("attention.attention.", "").replace("attention.output.", "")
    #endregion


def cosine(a, b):
    """
    * what? - Cosine similarity between two vectors, as a float.
    * why?  - It is the agreement measure between two trainings.
    * how?  - Dot product over the product of the norms.
    """
    #region 1 - Compute the cosine
    # what? - Normalized dot product.
    # how?  - .item() returns a plain Python float.
    return (a @ b / (a.norm() * b.norm())).item()
    #endregion


def main():
    """
    * what? - Computes the pairwise agreement for every matrix, writes the stage JSON and prints a summary.
    * why?  - This is the reproducibility evidence used by the aggregation and by the pages.
    * how?  - SVD of the original and of each trained matrix, sign alignment, 45 pairwise cosines averaged per matrix.
    """
    #region 1 - Parse flags and load the models
    # what? - Reads the stage and loads the original and the 10 trained models.
    # how?  - Names are sorted by block number and then by name.
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, default=1)
    stage = ap.parse_args().stage

    original = load_file("models/dinov2-small/model.safetensors")
    models = {s: load_file(stages.model(stage, s)) for s in SEEDS}
    names = sorted([n for n, t in original.items() if t.dim() == 2 and "encoder.layer." in n],
                   key=lambda n: (int(n.split(".")[2]), n))
    #endregion

    #region 2 - Measure the agreement per matrix
    # what? - For each matrix, the mean cosine over the 45 pairs of seeds, for weights and for directions.
    # how?  - torch.sign of the column-wise dot product resolves the sign ambiguity of U.
    output = []
    for n in names:
        U0, S0, _ = torch.linalg.svd(original[n].float(), full_matrices=False)
        dp, dd = {}, {}
        for s in SEEDS:
            U, S, _ = torch.linalg.svd(models[s][n].float(), full_matrices=False)
            sign = torch.sign((U[:, :TOP] * U0[:, :TOP]).sum(0))
            dd[s] = (U[:, :TOP] * sign - U0[:, :TOP]).flatten()
            dp[s] = (S[:TOP] - S0[:TOP]) / S0[:TOP]
        pairs = list(itertools.combinations(SEEDS, 2))
        output.append({"bloco": int(n.split(".")[2]), "tipo": kind(n),
                       "cos_pesos": st.mean(cosine(dp[i], dp[j]) for i, j in pairs),
                       "cos_direcoes": st.mean(cosine(dd[i], dd[j]) for i, j in pairs)})
    #endregion

    #region 3 - Write the JSON
    # what? - Saves the per-matrix agreement.
    # how?  - The path comes from stages.REPRODUCIBILITY.
    json.dump(output, open(stages.REPRODUCIBILITY[stage], "w"), indent=2)
    #endregion

    #region 4 - Print the summary
    # what? - Global statistics, per type and per block.
    # how?  - Same lines as the original script, with English labels.
    for k in ("cos_pesos", "cos_direcoes"):
        v = [x[k] for x in output]
        print(f"{k:<14} mean {st.mean(v):.3f} | median {st.median(v):.3f} | min {min(v):.3f} | max {max(v):.3f} | positive {sum(x > 0 for x in v)} of {len(v)}")
    for t in sorted({x["tipo"] for x in output}):
        g = [x for x in output if x["tipo"] == t]
        print(f"  {t:<8} weights {st.mean(x['cos_pesos'] for x in g):.3f} | directions {st.mean(x['cos_direcoes'] for x in g):.3f}")
    print("per block, directions:", [round(st.mean(x["cos_direcoes"] for x in output if x["bloco"] == b), 2) for b in range(12)])
    print("per block, weights:   ", [round(st.mean(x["cos_pesos"] for x in output if x["bloco"] == b), 2) for b in range(12)])
    #endregion


if __name__ == "__main__":
    main()
