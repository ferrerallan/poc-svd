"""
* what? - STAGE 4: measures what each restricted fine-tuning did to the 72 matrices and checks the restrictions held.
* why?  - The restriction must be verified, not assumed: sigma-only keeps U, rotation-only keeps sigma.
* how?  - Per matrix and seed: displacement, sigma change (top 10 and all), principal subspace angles, preserved directions.

Sanity checks: so_sigma must have a subspace angle ~0 (U preserved) and so_rotacao must have dsigma ~0 (sigma preserved).
'completo' = stage 3 models (read-only). Output: etapa4/medidas_variantes.json
Run from the repository root:  python3 doutorado/experimento1/etapa4/scripts/measure_variants.py
"""
import json
import statistics as st
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config_etapa4 as C  # noqa: E402


def mean_angle(U0k, U1k):
    """
    * what? - Returns the mean principal angle (degrees) between two subspaces.
    * why?  - Subspace angles do not depend on vector order or sign.
    * how?  - arccos of the singular values of U0k' U1k, clamped to 1, averaged.
    """
    #region 1 - Compute the angle
    # what? - Mean angle in degrees.
    # how?  - svdvals of the overlap matrix.
    return torch.rad2deg(torch.arccos(torch.linalg.svdvals(U0k.T @ U1k).clamp(max=1))).mean().item()
    #endregion


def main():
    """
    * what? - Measures the four variants over the available seeds and writes medidas_variantes.json.
    * why?  - It is the table that proves each restricted variant did only what it was allowed to do.
    * how?  - SVD of each original matrix once; per variant and seed the same measures as the ruler of stage 3.
    """
    #region 1 - Load the original model and its SVDs
    # what? - Original weights and the SVD of each of the 72 matrices.
    # how?  - The SVDs are reused for every variant and seed.
    original = load_file(str(C.ORIGINAL_MODEL / "model.safetensors"))
    names = sorted([n for n, t in original.items() if t.dim() == 2 and "encoder.layer." in n], key=lambda n: (int(n.split(".")[2]), n))
    svd0 = {n: torch.linalg.svd(original[n].float(), full_matrices=False) for n in names}
    output = {}
    #endregion

    #region 2 - Measure every variant
    # what? - Per-seed means over the 72 matrices, then statistics over the seeds.
    # how?  - Variants whose models do not exist are skipped.
    for variant in ["completo"] + C.VARIANTS:
        accum = {k: [] for k in ("desloc_pct", "sigma_top10_abs_pct", "sigma_todos_abs_pct", "sigma_top10_max_abs_pct", "sub10_graus", "sub50_graus", "direcoes_preservadas", "sigma_max_abs_pct")}
        reports = []
        for s in C.SEEDS:
            path = C.full_model(s) if variant == "completo" else C.variant_model(variant, s)
            if not path.exists():
                continue
            W1s = load_file(str(path))
            reports.append(json.load(open(path.parent / "relatorio.json")))
            per = {k: [] for k in accum}
            for n in names:
                W0 = original[n].float()
                W1 = W1s[n].float()
                U0, S0, _ = svd0[n]
                U1, S1, _ = torch.linalg.svd(W1, full_matrices=False)
                rel = (S1 - S0) / S0
                per["desloc_pct"].append(100 * ((W1 - W0).norm() / W0.norm()).item())
                per["sigma_top10_abs_pct"].append(100 * rel[:10].abs().mean().item())
                per["sigma_todos_abs_pct"].append(100 * rel.abs().mean().item())
                per["sigma_top10_max_abs_pct"].append(100 * rel[:10].abs().max().item())
                per["sub10_graus"].append(mean_angle(U0[:, :10], U1[:, :10]))
                per["sub50_graus"].append(mean_angle(U0[:, :50], U1[:, :50]))
                # sanity: is each new direction (up to sign and order) an old direction? 1.0 = the set of directions is preserved
                per["direcoes_preservadas"].append((U1.T @ U0).abs().max(dim=1).values.mean().item())
                per["sigma_max_abs_pct"].append(100 * rel.abs().max().item())
            for k in accum:
                accum[k].append(st.mean(per[k]))
        if not reports:
            continue
        acc = [r["acuracia_depois"] for r in reports]
        output[variant] = {"n_modelos": len(reports), "acuracia": {"media": st.mean(acc), "dp": st.stdev(acc) if len(acc) > 1 else 0, "por_semente": acc},
                           "acuracia_treino_final": st.mean(r["acuracia_treino_por_epoca"][-1] for r in reports),
                           "params_restritos": reports[0].get("params_restritos"), "lr_restrito": reports[0].get("lr_restrito"), "minutos": st.mean(r["minutos_de_treino"] for r in reports),
                           **{k: {"media": st.mean(v), "dp": st.stdev(v) if len(v) > 1 else 0} for k, v in accum.items()}}
        g = output[variant]
        print(f"{variant:<13} n={len(reports)} acc {g['acuracia']['media']:.2f}+-{g['acuracia']['dp']:.2f} | displacement {g['desloc_pct']['media']:.3f}% | s10 {g['sigma_top10_abs_pct']['media']:.4f}% (max {g['sigma_top10_max_abs_pct']['media']:.3f}%) | sAll {g['sigma_todos_abs_pct']['media']:.4f}% | sub10 {g['sub10_graus']['media']:.3f} | sub50 {g['sub50_graus']['media']:.3f} | preserved directions {g['direcoes_preservadas']['media']:.4f} | sigma max {g['sigma_max_abs_pct']['media']:.3f}% | restricted params {g['params_restritos']}")
    #endregion

    #region 3 - Write the result
    # what? - Saves medidas_variantes.json.
    # how?  - The keys stay in Portuguese: the page builder reads this schema.
    (C.STAGE4 / "medidas_variantes.json").write_text(json.dumps(output, indent=1))
    #endregion


if __name__ == "__main__":
    main()
