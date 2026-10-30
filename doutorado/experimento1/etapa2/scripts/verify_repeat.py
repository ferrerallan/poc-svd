"""
* what? - Checks the reproducibility of the procedure: does the same seed, trained twice, give exactly the same model?
* why?  - If not, we need to know how far apart the two runs are, relative to the change fine-tuning causes.
* how?  - Compares models/etapa2/s0 and models/etapa2/s0b tensor by tensor and relative to the training displacement.

Output: etapa2/repeticao.json
Run from the repository root:  python3 doutorado/experimento1/etapa2/scripts/verify_repeat.py
"""
import json
import statistics as st

import torch
from safetensors.torch import load_file


def main():
    """
    * what? - Measures the difference between two runs of seed 0 and writes repeticao.json.
    * why?  - The result shows that training is reproducible in outcome but not bit-for-bit deterministic.
    * how?  - Counts identical tensors, takes the largest absolute difference and the relative difference per matrix.
    """
    #region 1 - Load the models and the reports
    # what? - The two runs, the original model and both training reports.
    # how?  - s0 is the first run and s0b the repetition.
    a = load_file("models/etapa2/s0/model.safetensors")
    b = load_file("models/etapa2/s0b/model.safetensors")
    o = load_file("models/dinov2-small/model.safetensors")
    ra = json.load(open("models/etapa2/s0/relatorio.json"))
    rb = json.load(open("models/etapa2/s0b/relatorio.json"))
    #endregion

    #region 2 - Measure the differences
    # what? - Identical tensors, maximum difference and relative difference over the matrices.
    # how?  - Relative to ||W_a - W_original||, the size of the fine-tuning change.
    identical = sum(torch.equal(a[n], b[n]) for n in a)
    max_diff = max((a[n].float() - b[n].float()).abs().max().item() for n in a)
    mats = [n for n in a if a[n].dim() == 2 and "encoder.layer." in n]
    rel = [((a[n].float() - b[n].float()).norm() / (a[n].float() - o[n].float()).norm()).item() for n in mats]
    #endregion

    #region 3 - Write the result
    # what? - Saves the measures and the accuracies of both runs.
    # how?  - The keys stay in Portuguese: build_diary.py reads this schema.
    out = {"tensores_total": len(a), "tensores_identicos": identical, "diferenca_maxima_absoluta": max_diff,
           "diferenca_relativa_a_mudanca_do_treino_media": st.mean(rel),
           "acuracia_execucao_1": ra["acuracia_depois"], "acuracia_execucao_2": rb["acuracia_depois"],
           "acuracias_por_epoca_1": ra.get("acuracias_por_epoca"), "acuracias_por_epoca_2": rb.get("acuracias_por_epoca")}
    json.dump(out, open("doutorado/experimento1/etapa2/repeticao.json", "w"), indent=2)
    print(json.dumps(out, indent=2))
    #endregion


if __name__ == "__main__":
    main()
