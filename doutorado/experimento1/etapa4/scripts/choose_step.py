"""
* what? - STAGE 4: chooses the step (lr) of each restricted variant by the HIGHEST VALIDATION accuracy of the last epoch.
* why?  - The test set must not take part in the choice; only the pilots' validation split decides.
* how?  - Reads the pilot reports of each variant and lr, takes the best one and records all pilots.

Output: etapa4/passos_escolhidos.json
Run from the repository root:  python3 doutorado/experimento1/etapa4/scripts/choose_step.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config_etapa4 as C  # noqa: E402


def main():
    """
    * what? - Picks the best step per variant and writes passos_escolhidos.json.
    * why?  - The chosen steps feed the 10-seed trainings of stage 4.
    * how?  - For each variant and lr reads relatorio.json of the pilot, then takes the maximum final validation accuracy.
    """
    #region 1 - Read the pilots and choose
    # what? - One entry per variant with its pilots and the chosen lr.
    # how?  - The keys stay in Portuguese: the page builder and the recipe read this schema.
    out = {}
    for variant, lrs in C.LR_GRID.items():
        rows = []
        for lr in lrs:
            r = json.load(open(C.MODELS / f"{C.PILOT_PREFIX[variant]}{lr}" / "relatorio.json"))
            rows.append({"lr": float(lr), "lr_str": lr, "validacao_final": r["acuracia_depois"], "validacao_por_epoca": r["acuracias_por_epoca"], "treino_final": r["acuracia_treino_por_epoca"][-1]})
        best = max(rows, key=lambda x: x["validacao_final"])
        out[variant] = {"lr": best["lr_str"], "pilotos": rows}
        print(variant, "->", best["lr"], [(row["lr"], row["validacao_final"]) for row in rows])
    #endregion

    #region 2 - Write the result
    # what? - Saves the JSON.
    # how?  - indent 2, as in the frozen file.
    (C.STAGE4 / "passos_escolhidos.json").write_text(json.dumps(out, indent=2))
    #endregion


if __name__ == "__main__":
    main()
