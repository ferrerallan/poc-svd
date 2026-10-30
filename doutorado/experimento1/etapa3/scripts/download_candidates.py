"""
* what? - STAGE 3, step 0b: downloads the 4 candidate datasets (parquet, HuggingFace) and inspects columns and sizes.
* why?  - The choice of a distant domain needs the data on disk and a record of what each candidate contains.
* how?  - hf_hub_download of every parquet file of each repo into etapa3/dados/<nickname>/, then reads the first image.

Output: etapa3/dados/candidatos_inspecao.json (and the dataset README when it exists).
This script needs the network. Run from the repository root:
    python3 doutorado/experimento1/etapa3/scripts/download_candidates.py
"""
import io
import json
import sys
import time
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download, list_repo_files
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config_etapa3 as C  # noqa: E402

REPOS = {"dtd": "tanganke/dtd", "eurosat": "blanchon/EuroSAT_RGB", "pele": "Pranavkpba2000/skin_cancer_small_dataset", "kmnist": "tanganke/kmnist"}


def main():
    """
    * what? - Downloads each candidate dataset and writes the inspection summary.
    * why?  - The inspection file documents rows, columns, classes and image size of every candidate.
    * how?  - Per repo: list parquet files, download them, try the README, read the first image of each part.
    """
    #region 1 - Download and inspect every candidate
    # what? - Fills the summary dictionary.
    # how?  - Parts are named by the file name (train or test) and stored under the frozen JSON keys of the schema.
    summary = {}
    for nickname, repo in REPOS.items():
        t = time.time()
        files = [f for f in list_repo_files(repo, repo_type="dataset") if f.endswith(".parquet")]
        local = {("treino" if "train" in Path(f).name else "teste" if "test" in Path(f).name else Path(f).stem):
                 hf_hub_download(repo, f, repo_type="dataset", local_dir=str(C.DATA / nickname)) for f in files}
        for extra in ("README.md",):
            try:
                hf_hub_download(repo, extra, repo_type="dataset", local_dir=str(C.DATA / nickname))
            except Exception:
                pass
        info = {"repo": repo, "arquivos": {k: str(Path(v).relative_to(C.DATA)) for k, v in local.items()}, "segundos": round(time.time() - t)}
        for part in ("treino", "teste"):
            df = pd.read_parquet(local[part])
            v = df["image"].iloc[0]
            im = Image.open(io.BytesIO(v["bytes"] if isinstance(v, dict) else v))
            info[part] = {"linhas": len(df), "colunas": list(df.columns), "classes": int(df["label"].nunique()), "imagem": f"{im.size[0]}x{im.size[1]} {im.mode}"}
        summary[nickname] = info
        print(nickname, json.dumps({k: v for k, v in info.items() if k != "arquivos"}))
    #endregion

    #region 2 - Write the summary
    # what? - Saves candidatos_inspecao.json.
    # how?  - ensure_ascii=False keeps accents readable.
    json.dump(summary, open(C.DATA / "candidatos_inspecao.json", "w"), indent=2, ensure_ascii=False)
    #endregion


if __name__ == "__main__":
    main()
