"""
* what? - STAGE 3: FUNCTIONAL TEST by transplant; puts parts of the weight change back into the original matrices.
* why?  - Shows which part of the change carries the function: the sigma part or the directions part.
* how?  - For each fine-tuned model, replaces the 72 matrices by five variants and measures test accuracy of each.

For each fine-tuned model it puts back in the 72 matrices:
  completo     W1 (the fine-tuned model itself)
  so_sigma     W0 + D, where D = U0 diag(diag(U0^T dW V0)) V0^T  (only the "singular values" part of the change)
  so_direcoes  W0 + (dW - D)                                       (everything that is not the sigma part)
  direcoes_eq  W0 + (dW - D) * ||D||/||dW - D||                    (directions part with the SAME energy as the sigma part)
  nenhuma      W0                                                  (the 72 original matrices; the rest fine-tuned)
All other parameters (bias, LayerNorm, embeddings) and the head are those of the fine-tuned model.
Output: etapa3/transplante_<set>.json
Run from the repository root:  python3 doutorado/experimento1/etapa3/scripts/transplant.py --set etapa2|etapa3 [--seeds 0 1 ...]
"""
import argparse
import json
import statistics as st
import sys
import types
from pathlib import Path

import torch
from safetensors.torch import load_file
from torch.utils.data import DataLoader
from transformers import Dinov2Model

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config_etapa3 as C  # noqa: E402
import train_etapa3 as T  # noqa: E402

DEVICE = "cuda"


def test_loader_of(model_set, report):
    """
    * what? - Builds the test DataLoader of the dataset a model set was trained on, and returns it with the class count.
    * why?  - Stage 2 models were trained on pets and stage 3 models on skin lesions; each needs its own test data.
    * how?  - Builds an argument namespace like the training script's and calls train_etapa3.build_data.
    """
    #region 1 - Build the data arguments
    # what? - Paths and options of the right dataset.
    # how?  - Stage 2 uses the Oxford-IIIT Pet parquet files; stage 3 reads them from the training report.
    if model_set == "etapa2":
        ns = types.SimpleNamespace(train=str(C.ROOT / "data/hf/data/train-00000-of-00001.parquet"), test=str(C.ROOT / "data/hf/data/test-00000-of-00001.parquet"),
                                   image_col="image", label_col="label", max_train=C.MAX_TRAIN, max_test=C.MAX_TEST, shuffle_labels=False)
    else:
        ns = types.SimpleNamespace(train=report["treino_parquet"], test=report["teste_parquet"], image_col=report["col_imagem"], label_col=report["col_rotulo"],
                                   max_train=report["max_treino"], max_test=report["max_teste"], shuffle_labels=False)
    #endregion

    #region 2 - Build the loader
    # what? - Test loader and number of classes.
    # how?  - Only the test dataset is used; train and the class list come back from build_data.
    _, test, classes = T.build_data(ns)
    return DataLoader(test, batch_size=64, shuffle=False, num_workers=4, pin_memory=True), len(classes)
    #endregion


@torch.no_grad()
def head_accuracy(body, head, loader):
    """
    * what? - Returns the test accuracy of a body plus a linear head over a loader.
    * why?  - The transplant swaps weights inside the body, so body and head are handled separately.
    * how?  - Eval mode, bf16 autocast, head applied to the CLS token, argmax against the labels.
    """
    #region 1 - Count the hits
    # what? - Accumulates hits and total.
    # how?  - Token 0 of the last hidden state is the CLS.
    body.eval()
    hits = total = 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            hits += (head(body(pixel_values=x).last_hidden_state[:, 0]).argmax(1) == y).sum().item()
            total += y.numel()
    #endregion

    #region 2 - Return the percentage
    # what? - Converts counts to a percentage.
    # how?  - 100 * hits / total.
    return 100 * hits / total
    #endregion


def main():
    """
    * what? - Runs the five-variant transplant for each seed and writes transplante_<set>.json.
    * why?  - The functional test shows that the sigma part alone recovers none of the gain, while directions recover it.
    * how?  - Per seed: load the fine-tuned model and head, build the variants per matrix, load each into the body and test.
    """
    #region 1 - Parse flags and load the original model
    # what? - Reads the model set and seeds, loads W0 and lists the 72 matrix names.
    # how?  - Names are the 2-D tensors of the 12 blocks.
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True, choices=["etapa2", "etapa3"], dest="model_set")
    ap.add_argument("--seeds", type=int, nargs="*", default=C.SEEDS)
    a = ap.parse_args()
    original = load_file(str(C.ORIGINAL_MODEL / "model.safetensors"))
    names = [n for n, t in original.items() if t.dim() == 2 and "encoder.layer." in n]
    #endregion

    #region 2 - Transplant seed by seed
    # what? - For each seed, builds the variants and measures the accuracy of each.
    # how?  - D is the sigma part of dW and O the rest; "direcoes_eq" rescales O to the energy of D.
    output = {"conjunto": a.model_set, "variantes": ["completo", "so_sigma", "so_direcoes", "direcoes_eq", "nenhuma"], "sementes": []}
    loader = n_classes = None
    for s in a.seeds:
        folder = C.stage_model(int(a.model_set[-1]), s).parent
        report = json.load(open(folder / "relatorio.json"))
        if loader is None:
            loader, n_classes = test_loader_of(a.model_set, report)
        fine_tuned = load_file(str(folder / "model.safetensors"))
        head = torch.nn.Linear(384, n_classes).to(DEVICE)
        head.load_state_dict(torch.load(folder / "cabeca.pt", map_location=DEVICE))
        head.eval()
        body = Dinov2Model.from_pretrained(str(folder)).to(DEVICE)
        variants = {v: {} for v in output["variantes"]}
        fraction = []
        for n in names:
            W0, W1 = original[n].float(), fine_tuned[n].float()
            dW = W1 - W0
            U0, S0, Vt0 = torch.linalg.svd(W0, full_matrices=False)
            d = ((U0.T @ dW) * Vt0).sum(1)
            D = (U0 * d) @ Vt0
            O = dW - D
            fraction.append((D.norm() ** 2 / dW.norm() ** 2).item())
            variants["completo"][n] = W1
            variants["so_sigma"][n] = W0 + D
            variants["so_direcoes"][n] = W0 + O
            variants["direcoes_eq"][n] = W0 + O * (D.norm() / O.norm())
            variants["nenhuma"][n] = W0
        res = {}
        sd = body.state_dict()
        for v, mats in variants.items():
            for n, W in mats.items():
                sd[n] = W.to(sd[n].dtype)
            body.load_state_dict(sd)
            res[v] = round(head_accuracy(body, head, loader), 2)
        res["fracao_energia_sigma_media_pct"] = round(100 * sum(fraction) / len(fraction), 3)
        res["acuracia_relatorio"] = report["acuracia_depois"]
        res["semente"] = s
        output["sementes"].append(res)
        print(json.dumps(res), flush=True)
        del body, fine_tuned, variants
        torch.cuda.empty_cache()
    #endregion

    #region 3 - Summarize and write
    # what? - Mean and standard deviation of each variant over the seeds.
    # how?  - The keys stay in Portuguese: the page builder reads this schema.
    output["media"] = {v: st.mean(r[v] for r in output["sementes"]) for v in output["variantes"]}
    output["dp"] = {v: (st.stdev(r[v] for r in output["sementes"]) if len(output["sementes"]) > 1 else 0) for v in output["variantes"]}
    (C.STAGE3 / f"transplante_{a.model_set}.json").write_text(json.dumps(output, indent=1))
    print("mean:", json.dumps(output["media"]))
    #endregion


if __name__ == "__main__":
    main()
