"""
* what? - STAGE 8 (session 19): summarizes the four variants of DINOv2-Base on skin next to the DINOv2-Small numbers, and checks that the restrictions held.
* why?  - Session 19 needs one table per model with accuracy, trainable numbers and matrix measures, to see whether the Small ordering repeats on a wider model.
* how?  - Reads relatorio.json of the 40 Base models, measures their 72 matrices against models/dinov2-base as stage 5 did, and copies the Small skin summary of stage 5 for the side-by-side table.

Output: etapa8/resumo_etapa8.json
Run from the repository root:  python3 doutorado/experimento1/etapa8/scripts/summarize_etapa8.py
"""
import json
import statistics as st
import sys
from pathlib import Path

from safetensors.torch import load_file

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "etapa5" / "scripts"))
import summarize_etapa5 as S5  # noqa: E402

ROOT = HERE.parents[3]
STAGE8 = ROOT / "doutorado/experimento1/etapa8"
ORIGINAL = ROOT / "models/dinov2-base/model.safetensors"
SEEDS = list(range(10))
GROUPS = {"congelado_72": "models/etapa8/congelado_72_s{s}", "so_sigma": "models/etapa8/so_sigma_s{s}",
          "so_rotacao_bloco3": "models/etapa8/so_rotacao_bloco3_s{s}", "completo": "models/etapa8/completo_s{s}"}


def main():
    """
    * what? - Builds the Base summary, attaches the Small summary and the Base triage, and writes the JSON.
    * why?  - One file feeds the tables of session 19.
    * how?  - Same measures as stage 5 (displacement, top-10 sigma change, preserved directions), plus recovery fractions and the paired rotation-minus-sigma difference.
    """
    #region 1 - Original matrices of the Base
    # what? - W0 of the 72 matrices and their names.
    # how?  - 2-D tensors of the encoder layers, as in stage 5, but read from models/dinov2-base.
    original = load_file(str(ORIGINAL))
    names = [n for n, t in original.items() if t.dim() == 2 and "encoder.layer." in n]
    #endregion

    #region 2 - Summarize each group
    # what? - Accuracy statistics, trainable numbers, matrix measures.
    # how?  - Groups with missing models are skipped and reported.
    out = {"modelo": "dinov2-base", "dominio": "pele", "n_matrizes": len(names), "params_72_matrizes": sum(t.numel() for n, t in original.items() if n in names)}
    triage = STAGE8 / "triagem_base.json"
    if triage.exists():
        out["corpo_congelado_triagem"] = json.load(open(triage))["acuracia_corpo_congelado"]
    for group, pattern in GROUPS.items():
        folders = [ROOT / pattern.format(s=s) for s in SEEDS]
        if not all((f / "relatorio.json").exists() for f in folders):
            print(group, "skipped (models missing)")
            continue
        reports = [json.load(open(f / "relatorio.json")) for f in folders]
        acc = [r["acuracia_depois"] for r in reports]
        measures = [S5.matrix_measures(original, load_file(str(f / "model.safetensors")), names) for f in folders]
        out[group] = {"acuracia": {"media": st.mean(acc), "dp": st.stdev(acc), "min": min(acc), "max": max(acc), "por_semente": acc},
                      "acuracia_treino_final": st.mean(r["acuracia_treino_por_epoca"][-1] for r in reports),
                      "params_restritos_efetivos": reports[0].get("params_restritos_efetivos", reports[0].get("params_restritos")),
                      "params_resto_corpo": reports[0].get("params_resto_corpo"), "params_cabeca": reports[0].get("params_cabeca"),
                      "lr_restrito": reports[0].get("lr_restrito"), "minutos": st.mean(r["minutos_de_treino"] for r in reports),
                      "desloc_pct": st.mean(m[0] for m in measures), "sigma_top10_abs_pct": st.mean(m[1] for m in measures),
                      "direcoes_preservadas": st.mean(m[2] for m in measures)}
        g = out[group]
        print(f"{group:<18} acc {g['acuracia']['media']:.2f} +- {g['acuracia']['dp']:.2f} | params {g['params_restritos_efetivos']} | lr {g['lr_restrito']} | "
              f"disp {g['desloc_pct']:.3f}% | s10 {g['sigma_top10_abs_pct']:.5f}% | preserved {g['direcoes_preservadas']:.4f} | {g['minutos']:.1f} min", flush=True)
    #endregion

    #region 3 - Recovery, paired comparison, Small reference and write
    # what? - Fraction of the frozen-to-full gain each variant recovers; per-seed rotation minus sigma; the Small skin summary copied from stage 5.
    # how?  - Same definitions as sessions 13, 16 and 18.
    if all(k in out for k in GROUPS):
        gain = out["completo"]["acuracia"]["media"] - out["congelado_72"]["acuracia"]["media"]
        out["ganho_total"] = gain
        for k in ("so_sigma", "so_rotacao_bloco3"):
            out[k]["ganho_recuperado_pct"] = 100 * (out[k]["acuracia"]["media"] - out["congelado_72"]["acuracia"]["media"]) / gain
        a, b = out["so_rotacao_bloco3"]["acuracia"]["por_semente"], out["so_sigma"]["acuracia"]["por_semente"]
        out["rotacao_bloco3_menos_so_sigma"] = {"media": st.mean(x - y for x, y in zip(a, b)), "sementes_rotacao_acima": sum(x > y for x, y in zip(a, b))}
        c = out["completo"]["acuracia"]["por_semente"]
        out["completo_menos_so_sigma"] = {"media": st.mean(x - y for x, y in zip(c, b)), "sementes_completo_acima": sum(x > y for x, y in zip(c, b))}
        print("rotation(3) - sigma:", out["rotacao_bloco3_menos_so_sigma"], "| full - sigma:", out["completo_menos_so_sigma"])
    small = ROOT / "doutorado/experimento1/etapa5/resumo_etapa5.json"
    if small.exists():
        out["small_referencia"] = json.load(open(small))
    (STAGE8 / "resumo_etapa8.json").write_text(json.dumps(out, indent=1))
    print("written: doutorado/experimento1/etapa8/resumo_etapa8.json")
    #endregion


if __name__ == "__main__":
    main()
