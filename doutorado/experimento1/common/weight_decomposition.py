"""
* what? - Splits the change of the 10 largest singular values (the WEIGHTS) into a common and a differential part.
* why?  - Tests whether sigma moved as one uniform scale (the optimizer's weight decay) or per direction (learning).
* how?  - Relative change per sigma; its mean is the common part, the rest is differential; compares with decay alone.

  common:       all sigmas changing together, in the same proportion (one overall scale)
  differential: each sigma changing differently from the others (re-weighting the directions)

It also computes the shrinkage that AdamW weight decay alone would produce in this training.
Run from the repository root:  python3 doutorado/experimento1/common/weight_decomposition.py --stage 1|2
"""
import argparse
import json
import math
import statistics as st

import stages
import torch
from safetensors.torch import load_file

TOP = 10


def percent(x):
    """
    * what? - Converts a fraction to a percentage rounded to 5 decimals.
    * why?  - Every figure of decomposicao.json is stored as a percentage with that precision.
    * how?  - 100 * x, then round.
    """
    #region 1 - Convert
    # what? - Scales and rounds.
    # how?  - round(..., 5) matches the frozen JSON files.
    return round(100 * x, 5)
    #endregion


def main():
    """
    * what? - Computes the common/differential decomposition for one stage and writes decomposicao.json.
    * why?  - Separates what the optimizer does to every sigma from what the task asks of each direction.
    * how?  - Loops over matrices and seeds, then adds the expected weight-decay shrinkage from the training report.
    """
    #region 1 - Parse flags and load the models
    # what? - Reads the stage and loads the original model and the 10 fine-tuned models.
    # how?  - Matrices are the 2-D tensors of the 12 blocks.
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, default=1)
    args = ap.parse_args()
    stage = args.stage

    original = load_file("models/dinov2-small/model.safetensors")
    models = {s: load_file(stages.model(stage, s)) for s in stages.SEEDS}
    names = [n for n, t in original.items() if t.dim() == 2 and "encoder.layer." in n]
    #endregion

    #region 2 - Decompose every matrix and seed
    # what? - Fills one list per statistic.
    # how?  - r is the signed relative change of the 10 top sigmas; its mean c is the common part.
    total_abs, common_abs, diff_abs, common_sign, common_fraction, same_direction, consistency = [], [], [], [], [], [], []
    for n in names:
        s0 = torch.linalg.svdvals(original[n].float())[:TOP]
        signs = []
        for s in stages.SEEDS:
            s1 = torch.linalg.svdvals(models[s][n].float())[:TOP]
            r = (s1 - s0) / s0
            c = r.mean()
            total_abs.append(r.abs().mean().item())
            common_abs.append(abs(c).item())
            diff_abs.append((r - c).abs().mean().item())
            common_sign.append(c.item())
            common_fraction.append((TOP * c ** 2 / (r ** 2).sum()).item())
            same_direction.append(bool((r > 0).all() or (r < 0).all()))
            signs.append(c.item() > 0)
        consistency.append(max(sum(signs), len(signs) - sum(signs)))
    #endregion

    #region 3 - Compute the shrinkage expected from weight decay alone
    # what? - Product of the per-step decay factors over the training steps where the body is free.
    # how?  - Decoupled AdamW: p <- p * (1 - lr * wd) each step, with the cosine schedule; warm-up steps are skipped.
    report = json.load(open(stages.model_dir(stage, 0) / "relatorio.json"))
    wd, lr = stages.WEIGHT_DECAY[stage], report["lr_corpo"]
    per_epoch = 3680 // 32
    T = report["epocas"] * per_epoch
    factor = 1.0
    for t in range(report["epocas_aquecer"] * per_epoch, T):
        factor *= 1 - lr * (1 + math.cos(math.pi * t / T)) / 2 * wd
    #endregion

    #region 4 - Assemble and write the result
    # what? - Builds the output dictionary and saves it.
    # how?  - The keys stay in Portuguese: they are read by the aggregation and page builders.
    out = {"etapa": stage, "weight_decay": wd, "topo": TOP, "n_matrizes": len(names), "n_treinos": len(stages.SEEDS),
           "total_abs_pct": percent(st.mean(total_abs)),
           "comum_abs_pct": percent(st.mean(common_abs)), "diferencial_abs_pct": percent(st.mean(diff_abs)),
           "fracao_energia_comum_pct": round(100 * st.mean(common_fraction), 2),
           "mesmo_sentido_pct": round(100 * sum(same_direction) / len(same_direction), 2),
           "comum_media_com_sinal_pct": percent(st.mean(common_sign)),
           "comum_positiva_pct": round(100 * sum(c > 0 for c in common_sign) / len(common_sign), 2),
           "consistencia_media_de_10_treinos": round(st.mean(consistency), 2),
           "encolhimento_esperado_weight_decay_pct": percent(factor - 1)}
    (stages.STAGE_DIR[stage] / "decomposicao.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    #endregion


if __name__ == "__main__":
    main()
