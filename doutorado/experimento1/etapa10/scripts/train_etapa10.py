"""
* what? - STAGE 10 (session 21 of the diary): the stage 4 restricted training applied to a FOURTH domain, EuroSAT (satellite images, 10 classes), without changing train_etapa4.py.
* why?  - Session 20 repeats the closed protocol (frozen 72, sigma only, small-block rotation, full) on a fourth domain, the second domain where the frozen body already does well; the stage 4 script is kept intact so its old results stay traceable.
* how?  - Points the stage 4 configuration to the EuroSAT parquet files, runs train_etapa4.main() with the same command line, then fixes the dataset and stage fields of the written report.

Run from the repository root, with the same flags as train_etapa4.py, e.g.:
    python3 doutorado/experimento1/etapa10/scripts/train_etapa10.py --variant so_sigma --restricted-lr 1e-2 --seed 0 --output models/etapa10/so_sigma_s0
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "etapa4" / "scripts"))
import config_etapa4 as C4  # noqa: E402
import train_etapa4 as T4  # noqa: E402

EUROSAT_TRAIN = C4.STAGE3 / "dados/eurosat/data/train-00000-of-00001.parquet"
EUROSAT_TEST = C4.STAGE3 / "dados/eurosat/data/test-00000-of-00001.parquet"


def main():
    """
    * what? - Redirects the data paths to EuroSAT, trains, and corrects the report.
    * why?  - train_etapa4.py reads the paths from its configuration module and writes "pele" in the report; both are fixed here.
    * how?  - Module attributes are changed before main() runs; the report is rewritten after it returns.
    """
    #region 1 - Point the stage 4 configuration to EuroSAT and train
    # what? - Same training, other photos.
    # how?  - main() reads C.SKIN_TRAIN / C.SKIN_TEST at call time, so changing the module attributes is enough.
    C4.SKIN_TRAIN, C4.SKIN_TEST = EUROSAT_TRAIN, EUROSAT_TEST
    T4.main()
    #endregion

    #region 2 - Correct the report
    # what? - dataset, stage and a note saying which script ran.
    # how?  - The output folder is the --output argument of the command line.
    output = Path(sys.argv[sys.argv.index("--output") + 1])
    report_path = output / "relatorio.json"
    report = json.load(open(report_path))
    report.update({"etapa": 10, "dataset": "eurosat", "nota": "treinado por etapa10/scripts/train_etapa10.py, que chama train_etapa4.py com os dados do EuroSAT"})
    report_path.write_text(json.dumps(report, indent=1, ensure_ascii=False))
    #endregion


if __name__ == "__main__":
    main()
