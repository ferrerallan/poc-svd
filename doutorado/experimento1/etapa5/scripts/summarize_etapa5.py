"""
* what? - STAGE 5 (session 16 of the diary): summarizes the new trainings next to the stage 4 variants and checks that the rotation kept every sigma.
* why?  - Session 16 needs one table: accuracy, trainable numbers and change of the matrices for sigma-only, the large rotation, the small rotation and the full training.
* how?  - Reads relatorio.json of every model, loads the 72 matrices of each and measures displacement, sigma change and preserved directions as measure_variants.py does.

Output: etapa5/resumo_etapa5.json
Run from the repository root:  python3 doutorado/experimento1/etapa5/scripts/summarize_etapa5.py
"""
import json
import statistics as st
from pathlib import Path

import torch
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[4]
STAGE5 = ROOT / "doutorado/experimento1/etapa5"
ORIGINAL = ROOT / "models/dinov2-small/model.safetensors"
SEEDS = list(range(10))
GROUPS = {"congelado_72": "models/etapa4/congelado_72_s{s}", "so_sigma": "models/etapa4/so_sigma_s{s}",
          "so_rotacao_bloco32": "models/etapa4/so_rotacao_s{s}", "so_rotacao_bloco3": "models/etapa5/so_rotacao_bloco3_s{s}",
          "completo": "models/etapa3/s{s}"}


def matrix_measures(original, trained, names):
    """
    * what? - Returns the mean displacement, the mean relative change of the 10 largest sigmas and the preserved-directions index over the 72 matrices of one model.
    * why?  - The same three numbers stage 4 used to check that each restriction held.
    * how?  - SVD of W0 and W1 per matrix; the formulas copy measure_variants.py.
    """
    #region 1 - Measure every matrix
    # what? - Three lists, one value per matrix, then their means.
    # how?  - rel is (S1 - S0) / S0; preserved directions is the mean over new directions of the largest |cosine| with an old one.
    disp, s10, pres = [], [], []
    for n in names:
        W0, W1 = original[n].float(), trained[n].float()
        U0, S0, _ = torch.linalg.svd(W0, full_matrices=False)
        U1, S1, _ = torch.linalg.svd(W1, full_matrices=False)
        disp.append(100 * ((W1 - W0).norm() / W0.norm()).item())
        s10.append(100 * ((S1 - S0) / S0)[:10].abs().mean().item())
        pres.append((U1.T @ U0).abs().max(dim=1).values.mean().item())
    return st.mean(disp), st.mean(s10), st.mean(pres)
    #endregion


def main():
    """
    * what? - Builds the summary of the five groups and writes the JSON.
    * why?  - One file feeds the table of session 16.
    * how?  - Per group: accuracies from the reports, trainable numbers from the first report, matrix measures averaged over the seeds.
    """
    #region 1 - Load the original matrices
    # what? - W0 of the 72 matrices.
    # how?  - 2-D tensors of the encoder layers.
    original = load_file(str(ORIGINAL))
    names = [n for n, t in original.items() if t.dim() == 2 and "encoder.layer." in n]
    #endregion

    #region 2 - Summarize each group
    # what? - Accuracy statistics, trainable numbers, step, minutes and matrix measures.
    # how?  - Groups whose models do not exist (sigma retrained with a new step) are skipped silently.
    out = {}
    if (STAGE5 / "passos_escolhidos.json").exists():
        steps = json.load(open(STAGE5 / "passos_escolhidos.json"))
        if steps["so_sigma"]["novo_passo_venceu"]:
            GROUPS["so_sigma_passo_novo"] = "models/etapa5/so_sigma_lr" + steps["so_sigma"]["lr"] + "_s{s}"
    for group, pattern in GROUPS.items():
        folders = [ROOT / pattern.format(s=s) for s in SEEDS]
        if not all((f / "relatorio.json").exists() for f in folders):
            print(group, "skipped (models missing)")
            continue
        reports = [json.load(open(f / "relatorio.json")) for f in folders]
        acc = [r["acuracia_depois"] for r in reports]
        measures = [matrix_measures(original, load_file(str(f / "model.safetensors")), names) for f in folders]
        out[group] = {"acuracia": {"media": st.mean(acc), "dp": st.stdev(acc), "min": min(acc), "max": max(acc), "por_semente": acc},
                      "acuracia_treino_final": st.mean(r["acuracia_treino_por_epoca"][-1] for r in reports),
                      "params_restritos_efetivos": reports[0].get("params_restritos_efetivos", reports[0].get("params_restritos")),
                      "lr_restrito": reports[0].get("lr_restrito"), "bloco_oft": reports[0].get("bloco_oft"),
                      "minutos": st.mean(r["minutos_de_treino"] for r in reports),
                      "desloc_pct": st.mean(m[0] for m in measures), "sigma_top10_abs_pct": st.mean(m[1] for m in measures),
                      "direcoes_preservadas": st.mean(m[2] for m in measures)}
        g = out[group]
        print(f"{group:<20} acc {g['acuracia']['media']:.2f} +- {g['acuracia']['dp']:.2f} | params {g['params_restritos_efetivos']} | lr {g['lr_restrito']} | "
              f"disp {g['desloc_pct']:.3f}% | s10 {g['sigma_top10_abs_pct']:.5f}% | preserved {g['direcoes_preservadas']:.4f} | {g['minutos']:.1f} min", flush=True)
    #endregion

    #region 3 - Paired comparison and write
    # what? - Per-seed difference between the small rotation and sigma-only, and the JSON.
    # how?  - Same seed numbers on both sides; counts how many seeds favour each.
    if "so_rotacao_bloco3" in out and "so_sigma" in out:
        a, b = out["so_rotacao_bloco3"]["acuracia"]["por_semente"], out["so_sigma"]["acuracia"]["por_semente"]
        out["rotacao_bloco3_menos_so_sigma"] = {"media": st.mean(x - y for x, y in zip(a, b)), "sementes_rotacao_acima": sum(x > y for x, y in zip(a, b))}
        print("rotation(3) - sigma:", out["rotacao_bloco3_menos_so_sigma"])
    (STAGE5 / "resumo_etapa5.json").write_text(json.dumps(out, indent=1))
    print("written: doutorado/experimento1/etapa5/resumo_etapa5.json")
    #endregion


if __name__ == "__main__":
    main()
