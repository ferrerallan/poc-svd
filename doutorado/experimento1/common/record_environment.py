"""
* what? - Records the environment an experiment ran in: versions, machine, git state and file hashes.
* why?  - Reproducibility: a stage result is only comparable if we know the exact code, model and data behind it.
* how?  - Collects package versions, GPU and driver, git commit and SHA-256 of scripts, original model and datasets.

Run from the repository root:
    python3 doutorado/experimento1/common/record_environment.py --output doutorado/experimento1/etapaN/ambiente.json
"""
import argparse
import datetime
import hashlib
import json
import platform
import subprocess
from pathlib import Path

import huggingface_hub
import numpy
import pandas
import PIL
import safetensors
import sklearn
import torch
import torchvision
import transformers

EXPERIMENT = Path("doutorado/experimento1")


def sha(path):
    """
    * what? - Returns the SHA-256 hex digest of a file.
    * why?  - Hashes identify the exact scripts, model and data used by a run.
    * how?  - Reads the file in 1 MiB chunks so large files do not fill memory.
    """
    #region 1 - Hash the file
    # what? - Streams the file through hashlib.
    # how?  - iter with a sentinel reads blocks until the empty bytes object.
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()
    #endregion


def shell(command):
    """
    * what? - Runs a shell command and returns its stripped standard output.
    * why?  - nvidia-smi and git give facts that Python packages do not expose.
    * how?  - subprocess.run with a shell; a missing tool just yields an empty string.
    """
    #region 1 - Run the command
    # what? - Captures the output text.
    # how?  - Errors are not raised: the caller records whatever came back.
    return subprocess.run(command, shell=True, capture_output=True, text=True).stdout.strip()
    #endregion


def script_hashes():
    """
    * what? - Hashes every Python script of experiment 1, in the shared folder and in each stage folder.
    * why?  - Scripts moved into common/ and etapa*/scripts/, so a single glob no longer finds them all.
    * how?  - Globs common/*.py and etapa*/scripts/*.py and keys the digests by path relative to the experiment.
    """
    #region 1 - Collect and hash
    # what? - Builds the relative path -> digest dictionary.
    # how?  - Sorted paths keep the JSON stable between runs.
    paths = sorted(list(EXPERIMENT.glob("common/*.py")) + list(EXPERIMENT.glob("etapa*/scripts/*.py")))
    return {str(p.relative_to(EXPERIMENT)): sha(p) for p in paths}
    #endregion


def main():
    """
    * what? - Writes the environment JSON and prints a short summary.
    * why?  - Every stage records its environment next to its results.
    * how?  - Builds one dictionary from versions, hardware, git and hashes and dumps it as JSON.
    """
    #region 1 - Parse flags
    # what? - Reads the output path.
    # how?  - Default points at the experiment root, as in the original script.
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="doutorado/experimento1/ambiente.json")
    args = ap.parse_args()
    #endregion

    #region 2 - Collect the facts
    # what? - Versions, machine, git state and hashes.
    # how?  - The keys stay in Portuguese: they are the schema of the frozen ambiente.json files.
    info = {
        "registrado_em": datetime.datetime.now().isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "sistema": platform.platform(),
        "pacotes": {m.__name__: m.__version__ for m in
                    (numpy, pandas, PIL, sklearn, torch, torchvision, transformers, safetensors, huggingface_hub)},
        "cuda_do_torch": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "driver_nvidia": shell("nvidia-smi --query-gpu=driver_version --format=csv,noheader"),
        "vram_total": shell("nvidia-smi --query-gpu=memory.total --format=csv,noheader"),
        "git_commit": shell("git rev-parse HEAD"),
        "git_arquivos_nao_versionados": shell("git status --porcelain").splitlines(),
        "sha256_scripts": script_hashes(),
        "sha256_modelo_original": sha("models/dinov2-small/model.safetensors"),
        "sha256_dados": {p.name: sha(p) for p in sorted(Path("data/hf/data").glob("*.parquet"))},
    }
    #endregion

    #region 3 - Write and report
    # what? - Saves the JSON and prints the main fields.
    # how?  - Creates the output folder first, in case the stage folder does not exist yet.
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(info, indent=2, ensure_ascii=False))
    print("environment recorded in", args.output)
    print(json.dumps({k: info[k] for k in ("python", "pacotes", "gpu", "driver_nvidia", "git_commit")}, indent=2))
    #endregion


if __name__ == "__main__":
    main()
