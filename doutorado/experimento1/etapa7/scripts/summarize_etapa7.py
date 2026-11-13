"""
* what? - STAGE 7 (session 18): summarizes the four variants on KMNIST next to the skin numbers, and checks that the restrictions held.
* why?  - Session 18 needs one table per domain with accuracy, trainable numbers and matrix measures, to see whether the skin ordering repeats.
* how?  - Reads relatorio.json of the 40 KMNIST models, measures the 72 matrices as stage 5 did, and copies the skin summary of stage 5 for the side-by-side table.

Output: etapa7/resumo_etapa7.json
Run from the repository root:  python3 doutorado/experimento1/etapa7/scripts/summarize_etapa7.py
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
STAGE7 = ROOT / "doutorado/experimento1/etapa7"
SEEDS = list(range(10))
GROUPS = {"congelado_72": "models/etapa7/congelado_72_s{s}", "so_sigma": "models/etapa7/so_sigma_s{s}",
          "so_rotacao_bloco3": "models/etapa7/so_rotacao_bloco3_s{s}", "completo": "models/etapa7/completo_s{s}"}
TRIAGE = json.load(open(ROOT / "doutorado/experimento1/etapa3/triagem.json"))["kmnist"]


def main():
    """
    * what? - Builds the KMNIST summary, attaches the skin summary, and writes the JSON.
    * why?  - One file feeds the tables of session 18.
    * how?  - Same measures as stage 5 (displacement, top-10 sigma change, preserved directions), plus recovery fractions and the paired rotation-minus-sigma difference.
    """
    #region 1 - Original matrices
    # what? - W0 of the 72 matrices and their names.
    # how?  - Reused from stage 5.
    original = load_file(str(S5.ORIGINAL))
    names = [n for n, t in original.items() if t.dim() == 2 and "encoder.layer." in n]
    #endregion

    #region 2 - Summarize each group
    # what? - Accuracy statistics, trainable numbers, matrix measures.
    # how?  - Groups with missing models are skipped and reported.
    out = {"dominio": "kmnist", "corpo_congelado_triagem": TRIAGE["acuracia_corpo_congelado"], "classes": TRIAGE["classes"], "acaso_pct": TRIAGE["acaso_pct"]}
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
                      "lr_restrito": reports[0].get("lr_restrito"), "minutos": st.mean(r["minutos_de_treino"] for r in reports),
                      "desloc_pct": st.mean(m[0] for m in measures), "sigma_top10_abs_pct": st.mean(m[1] for m in measures),
                      "direcoes_preservadas": st.mean(m[2] for m in measures)}
        g = out[group]
        print(f"{group:<18} acc {g['acuracia']['media']:.2f} +- {g['acuracia']['dp']:.2f} | params {g['params_restritos_efetivos']} | lr {g['lr_restrito']} | "
              f"disp {g['desloc_pct']:.3f}% | s10 {g['sigma_top10_abs_pct']:.5f}% | preserved {g['direcoes_preservadas']:.4f} | {g['minutos']:.1f} min", flush=True)
    #endregion

    #region 3 - Recovery, paired comparison, skin reference and write
    # what? - Fraction of the frozen-to-full gain each variant recovers; per-seed rotation minus sigma; the skin summary copied from stage 5.
    # how?  - Same definitions as sessions 13 and 16.
    if all(k in out for k in GROUPS):
        gain = out["completo"]["acuracia"]["media"] - out["congelado_72"]["acuracia"]["media"]
        out["ganho_total"] = gain
        for k in ("so_sigma", "so_rotacao_bloco3"):
            out[k]["ganho_recuperado_pct"] = 100 * (out[k]["acuracia"]["media"] - out["congelado_72"]["acuracia"]["media"]) / gain
        a, b = out["so_rotacao_bloco3"]["acuracia"]["por_semente"], out["so_sigma"]["acuracia"]["por_semente"]
        out["rotacao_bloco3_menos_so_sigma"] = {"media": st.mean(x - y for x, y in zip(a, b)), "sementes_rotacao_acima": sum(x > y for x, y in zip(a, b))}
        print("rotation(3) - sigma:", out["rotacao_bloco3_menos_so_sigma"])
    skin = ROOT / "doutorado/experimento1/etapa5/resumo_etapa5.json"
    if skin.exists():
        out["pele_referencia"] = json.load(open(skin))
    (STAGE7 / "resumo_etapa7.json").write_text(json.dumps(out, indent=1))
    print("written: doutorado/experimento1/etapa7/resumo_etapa7.json")
    #endregion


if __name__ == "__main__":
    main()
