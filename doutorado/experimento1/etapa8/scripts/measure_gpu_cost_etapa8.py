"""
* what? - STAGE 8 (session 19): the GPU cost measurement of session 14 (etapa4/scripts/measure_gpu_cost.py) applied to DINOv2-Base.
* why?  - Session 14 found that sigma-only saves memory and disk but not time on the Small; a model four times larger is where that saving should become visible.
* how?  - Points the stage 4 configuration to models/dinov2-base and the rotation to blocks of 3 (the variant trained in this stage), sets the chosen steps and the output file, then runs measure_gpu_cost.main() unchanged.

Run from the repository root:  python3 doutorado/experimento1/etapa8/scripts/measure_gpu_cost_etapa8.py
Output: etapa8/custo_gpu_base.json
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "etapa4" / "scripts"))
import config_etapa4 as C4  # noqa: E402
import measure_gpu_cost as G  # noqa: E402

STAGE8 = HERE.parent
BASE_MODEL = C4.ROOT / "models/dinov2-base"


def main():
    """
    * what? - Redirects model, rotation block, steps and output, then measures the four variants.
    * why?  - The measurement code must be the same as in session 14, so the two tables are comparable.
    * how?  - Module attributes are changed before main() runs; the steps come from etapa8/passos_escolhidos.json.
    """
    #region 1 - Redirect the configuration
    # what? - Base model, rotation of blocks of 3, chosen steps, stage 8 output file.
    # how?  - measure_gpu_cost reads C.ORIGINAL_MODEL, C.OFT_BLOCK, RESTRICTED_LR and OUTPUT at call time.
    steps = json.load(open(STAGE8 / "passos_escolhidos.json"))
    C4.ORIGINAL_MODEL = BASE_MODEL
    C4.OFT_BLOCK = 3
    G.RESTRICTED_LR = {"so_sigma": float(steps["so_sigma"]["lr"]), "so_rotacao": float(steps["so_rotacao_bloco3"]["lr"])}
    G.OUTPUT = STAGE8 / "custo_gpu_base.json"
    #endregion

    #region 2 - Measure
    # what? - Same four variants, same 30 steps, same preloaded batches.
    # how?  - Unchanged main() of the session 14 script.
    G.main()
    #endregion


if __name__ == "__main__":
    main()
