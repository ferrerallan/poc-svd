"""
* what? - STAGE 8 (session 19 of the diary): the stage 4 restricted training on skin lesions with a LARGER model, DINOv2-Base, without changing train_etapa4.py.
* why?  - Session 19 asks whether the skin findings (sigma only above the small rotation, most of the gain recovered) depend on the size of the model; the domain and the protocol stay the same, only the model changes.
* how?  - Points the stage 4 configuration to models/dinov2-base, runs train_etapa4.main() with the same command line, then fixes the stage and model fields of the written report.

Run from the repository root, with the same flags as train_etapa4.py, e.g.:
    python3 doutorado/experimento1/etapa8/scripts/train_etapa8.py --variant so_sigma --restricted-lr 1e-2 --seed 0 --output models/etapa8/so_sigma_s0
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "etapa4" / "scripts"))
import config_etapa4 as C4  # noqa: E402
import train_etapa4 as T4  # noqa: E402

BASE_MODEL = C4.ROOT / "models/dinov2-base"


def main():
    """
    * what? - Redirects the original model to DINOv2-Base, trains, and corrects the report.
    * why?  - train_etapa4.py reads the model path from its configuration module and writes stage 4 in the report; both are fixed here.
    * how?  - The module attribute is changed before main() runs; the report is rewritten after it returns.
    """
    #region 1 - Point the stage 4 configuration to DINOv2-Base and train
    # what? - Same training, same skin photos, larger body.
    # how?  - Classifier and materialized_state_dict read C.ORIGINAL_MODEL at call time, so changing the module attribute is enough.
    C4.ORIGINAL_MODEL = BASE_MODEL
    T4.main()
    #endregion

    #region 2 - Correct the report
    # what? - stage, model and a note saying which script ran.
    # how?  - The output folder is the --output argument of the command line.
    output = Path(sys.argv[sys.argv.index("--output") + 1])
    report_path = output / "relatorio.json"
    report = json.load(open(report_path))
    report.update({"etapa": 8, "modelo": "dinov2-base", "nota": "treinado por etapa8/scripts/train_etapa8.py, que chama train_etapa4.py com o modelo models/dinov2-base"})
    report_path.write_text(json.dumps(report, indent=1, ensure_ascii=False))
    #endregion


if __name__ == "__main__":
    main()
