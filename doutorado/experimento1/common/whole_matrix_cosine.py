"""
* what? - Agreement between trainings looking at the WHOLE matrix, not only the 10 largest sigmas or directions.
* why?  - Confirms with the full change dW that independent trainings do (or do not) move the weights alike.
* how?  - For each matrix and pair of seeds: cosine of (W_i - W0) and (W_j - W0) and the relative distance between them.

If the trainings were independent the cosine would be 0 and the relative distance 1.41.
Run from the repository root:  python3 doutorado/experimento1/common/whole_matrix_cosine.py --stage 1|2
"""
import argparse
import itertools
import json
import statistics as st

import stages
from safetensors.torch import load_file


def main():
    """
    * what? - Computes pairwise cosine and relative distance of the weight changes and writes cosseno_matriz.json.
    * why?  - It complements the top-10 analysis with the full-matrix view used in the pages.
    * how?  - Loads original and trained models, flattens each dW and averages over matrices and seed pairs.
    """
    #region 1 - Parse flags and load the models
    # what? - Reads the stage and loads the original and the trained models.
    # how?  - Matrices are the 2-D tensors of the 12 blocks.
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, default=1)
    stage = ap.parse_args().stage
    original = load_file("models/dinov2-small/model.safetensors")
    models = {s: load_file(stages.model(stage, s)) for s in stages.SEEDS}
    names = [n for n, t in original.items() if t.dim() == 2 and "encoder.layer." in n]
    #endregion

    #region 2 - Measure every matrix and pair of seeds
    # what? - Lists of cosines and relative distances.
    # how?  - dist = ||d_i - d_j|| / mean(||d_i||, ||d_j||).
    cos, dist = [], []
    for n in names:
        d = {s: (models[s][n].float() - original[n].float()).flatten() for s in stages.SEEDS}
        for i, j in itertools.combinations(stages.SEEDS, 2):
            cos.append((d[i] @ d[j] / (d[i].norm() * d[j].norm())).item())
            dist.append(((d[i] - d[j]).norm() / ((d[i].norm() + d[j].norm()) / 2)).item())
    #endregion

    #region 3 - Write the result
    # what? - Saves the summary statistics.
    # how?  - The keys stay in Portuguese: the page builders read this schema.
    out = {"etapa": stage, "pares": len(list(itertools.combinations(stages.SEEDS, 2))), "matrizes": len(names),
           "cosseno_medio": st.mean(cos), "cosseno_min": min(cos), "cosseno_max": max(cos),
           "distancia_relativa_media": st.mean(dist)}
    (stages.STAGE_DIR[stage] / "cosseno_matriz.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    #endregion


if __name__ == "__main__":
    main()
