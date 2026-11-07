"""
* what? - STAGE 5 (session 16 of the diary): chooses the step (lr) of the small-block rotation and decides whether a larger step helps sigma-only, by validation.
* why?  - The step is a free choice that must not look at the test set; the pilots' validation split decides, as in stage 4.
* how?  - Reads the pilot reports of this stage, adds the stage 4 sigma pilots (lr 1e-5 to 1e-2) to the sigma grid, and keeps the best final validation accuracy per variant.

Output: etapa5/passos_escolhidos.json
Run from the repository root:  python3 doutorado/experimento1/etapa5/scripts/choose_step_etapa5.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
MODELS4, MODELS5 = ROOT / "models/etapa4", ROOT / "models/etapa5"
STAGE5 = ROOT / "doutorado/experimento1/etapa5"
ROTATION_GRID = ["1e-5", "1e-4", "1e-3", "1e-2"]          # pilots of this stage: piloto_rotacao3_lr<lr>
SIGMA_OLD_GRID = ["1e-5", "1e-4", "1e-3", "1e-2"]         # stage 4 pilots: piloto_so_sigma_lr<lr>
SIGMA_NEW_GRID = ["3e-2", "1e-1"]                         # pilots of this stage: piloto_so_sigma_lr<lr>


def pilot_row(folder, lr):
    """
    * what? - Reads one pilot report and returns its summary row.
    * why?  - Every pilot is judged by the same two numbers: final validation accuracy and final train accuracy.
    * how?  - relatorio.json written by train_etapa4.py.
    """
    #region 1 - Read
    # what? - A small dictionary.
    # how?  - acuracia_depois is the validation accuracy when the pilot ran with --validation.
    r = json.load(open(folder / "relatorio.json"))
    return {"lr": float(lr), "lr_str": lr, "validacao_final": r["acuracia_depois"], "validacao_por_epoca": r["acuracias_por_epoca"],
            "treino_final": r["acuracia_treino_por_epoca"][-1], "pasta": str(folder.relative_to(ROOT))}
    #endregion


def main():
    """
    * what? - Chooses the steps and writes passos_escolhidos.json.
    * why?  - The final trainings of the stage read the chosen steps from this file.
    * how?  - Maximum of the final validation accuracy over the grid of each variant.
    """
    #region 1 - Collect the pilots and choose
    # what? - Rotation (3-line blocks) from this stage; sigma from stage 4 plus the two new steps.
    # how?  - The sigma choice also records whether a new step beat the stage 4 choice (1e-2).
    rot = [pilot_row(MODELS5 / f"piloto_rotacao3_lr{lr}", lr) for lr in ROTATION_GRID]
    sig = [pilot_row(MODELS4 / f"piloto_so_sigma_lr{lr}", lr) for lr in SIGMA_OLD_GRID] + [pilot_row(MODELS5 / f"piloto_so_sigma_lr{lr}", lr) for lr in SIGMA_NEW_GRID]
    best_rot, best_sig = max(rot, key=lambda x: x["validacao_final"]), max(sig, key=lambda x: x["validacao_final"])
    out = {"so_rotacao_bloco3": {"lr": best_rot["lr_str"], "pilotos": rot},
           "so_sigma": {"lr": best_sig["lr_str"], "lr_etapa4": "1e-2", "novo_passo_venceu": best_sig["lr_str"] in SIGMA_NEW_GRID, "pilotos": sig}}
    for k, v in out.items():
        print(k, "->", v["lr"], [(p["lr_str"], p["validacao_final"]) for p in v["pilotos"]])
    #endregion

    #region 2 - Write
    # what? - The JSON the recipe reads.
    # how?  - Keys in Portuguese to match passos_escolhidos.json of stage 4.
    (STAGE5 / "passos_escolhidos.json").write_text(json.dumps(out, indent=2))
    #endregion


if __name__ == "__main__":
    main()
