"""
* what? - STAGE 8 (session 19): chooses the step (lr) of sigma-only and of the small-block rotation for DINOv2-Base on skin, by validation.
* why?  - The steps chosen for the Small need not hold for a wider model; the pilots of this stage decide.
* how?  - Reads the pilot reports of this stage and keeps, per variant, the step with the best final validation accuracy.

Output: etapa8/passos_escolhidos.json
Run from the repository root:  python3 doutorado/experimento1/etapa8/scripts/choose_step_etapa8.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
MODELS = ROOT / "models/etapa8"
STAGE8 = ROOT / "doutorado/experimento1/etapa8"
GRID = {"so_sigma": ["1e-3", "1e-2", "3e-2", "1e-1"], "so_rotacao_bloco3": ["1e-4", "1e-3", "1e-2"]}


def main():
    """
    * what? - Picks the best step per variant and writes passos_escolhidos.json.
    * why?  - The final trainings of the stage read the chosen steps from this file.
    * how?  - Maximum of acuracia_depois (validation accuracy in a pilot) over the grid.
    """
    #region 1 - Read the pilots and choose
    # what? - One entry per variant with its pilots and the chosen lr.
    # how?  - Pilot folders are piloto_<variant>_lr<lr> inside models/etapa8.
    out = {}
    for variant, lrs in GRID.items():
        rows = []
        for lr in lrs:
            r = json.load(open(MODELS / f"piloto_{variant}_lr{lr}" / "relatorio.json"))
            rows.append({"lr": float(lr), "lr_str": lr, "validacao_final": r["acuracia_depois"], "validacao_por_epoca": r["acuracias_por_epoca"], "treino_final": r["acuracia_treino_por_epoca"][-1]})
        best = max(rows, key=lambda x: x["validacao_final"])
        out[variant] = {"lr": best["lr_str"], "pilotos": rows}
        print(variant, "->", best["lr_str"], [(p["lr_str"], p["validacao_final"]) for p in rows])
    #endregion

    #region 2 - Write
    # what? - The JSON the recipe reads.
    # how?  - Same schema as stages 5 and 7.
    (STAGE8 / "passos_escolhidos.json").write_text(json.dumps(out, indent=2))
    #endregion


if __name__ == "__main__":
    main()
