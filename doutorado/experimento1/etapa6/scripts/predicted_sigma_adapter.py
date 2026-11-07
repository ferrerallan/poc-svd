"""
* what? - STAGE 6 (session 17 of the diary): builds a sigma adapter from the footprint of FREE fine-tuning, without training any sigma, and measures it with the frozen-body protocol.
* why?  - Session 15 found that the free footprint on sigma points the same way as the learned sigma-only delta; this tests whether that footprint, amplified, already works as an adapter.
* how?  - Per seed: footprint = diag(U0^T dW V0) / S0 per matrix; predicted delta = k * footprint; W = U0 diag(S0 (1 + delta)) V0^T put into the ORIGINAL body (everything else at W0); CLS features extracted and a logistic regression fitted, as in the stage 3 screening. k is chosen by a validation split of the train set. The photos are decoded and transformed once and kept as float16 tensors, so each variant costs only forward passes.

Variants measured per seed, all with the rest of the body original and only the logistic regression fitted:
  base               W0 (the triage number of session 12, recomputed once)
  previsto_k<k>      the amplified footprint, for every k of the grid (validation decides k; test reported for all)
  aleatorio_k<k>     a Gaussian delta per matrix with the same norm as the predicted one (control), same ks
  aprendido          the delta learned by sigma-only training with step 3e-2 (session 16), put into the original body
Run from the repository root:
    python3 -u doutorado/experimento1/etapa6/scripts/predicted_sigma_adapter.py | tee doutorado/experimento1/etapa6/logs/adaptador_previsto.log
Output: doutorado/experimento1/etapa6/adaptador_previsto.json
"""
import json
import statistics as st
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from safetensors.torch import load_file
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from torchvision import transforms
from transformers import Dinov2Model

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "etapa3" / "scripts"))
import config_etapa3 as C3  # noqa: E402
import domain_screening as DS  # noqa: E402

ROOT = HERE.parents[3]
STAGE6 = ROOT / "doutorado/experimento1/etapa6"
ORIGINAL = ROOT / "models/dinov2-small"
SEEDS = list(range(10))
KS = [10, 33, 100]
VALIDATION_SEED, VALIDATION_FRACTION = 4242, 0.10
OUTPUT = STAGE6 / "adaptador_previsto.json"
DEVICE = "cuda"


def skin_tables():
    """
    * what? - Returns the train and test tables of the skin dataset, sub-sampled exactly as every stage 3 script does.
    * why?  - The adapter must be judged on the same photos as the triage and the trainings.
    * how?  - Same generator, same seed, train drawn first and test second.
    """
    #region 1 - Read and sub-sample
    # what? - Two dataframes.
    # how?  - Copies the sub-sampling lines of train_etapa3.build_data.
    c = DS.CANDIDATES["pele"]
    tr, te = pd.read_parquet(C3.DATA / c["treino"]), pd.read_parquet(C3.DATA / c["teste"])
    g = np.random.default_rng(C3.SUBSAMPLE_SEED)
    if len(tr) > C3.MAX_TRAIN:
        tr = tr.iloc[np.sort(g.choice(len(tr), C3.MAX_TRAIN, replace=False))]
    if len(te) > C3.MAX_TEST:
        te = te.iloc[np.sort(g.choice(len(te), C3.MAX_TEST, replace=False))]
    return tr, te
    #endregion


def cache_photos(transform, df):
    """
    * what? - Decodes and transforms every photo of a table once and returns a float16 tensor (N x 3 x 224 x 224).
    * why?  - Decoding JPEGs dominated the run time; each variant only needs forward passes.
    * how?  - DS.open_image and the test transform, stacked on the CPU in half precision (1.1 GB for 3,680 photos).
    """
    #region 1 - Decode once
    # what? - The stacked tensor.
    # how?  - float16 halves the memory; the model runs in bf16 autocast anyway.
    return torch.stack([transform(DS.open_image(v)) for v in df["image"].tolist()]).half()
    #endregion


def extract_cached(body, photos):
    """
    * what? - Returns the CLS features (N x 384) of a cached photo tensor.
    * why?  - Same numerics as DS.extract (batches of 64, bf16 autocast, token 0), without re-decoding the photos.
    * how?  - Batches moved to the GPU as float32, no_grad plus autocast.
    """
    #region 1 - Forward in batches
    # what? - A numpy array.
    # how?  - .float() restores float32 before the model, as the transform produced.
    feats = []
    for i in range(0, len(photos), 64):
        x = photos[i:i + 64].to(DEVICE).float()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            feats.append(body(pixel_values=x).last_hidden_state[:, 0].float().cpu().numpy())
    return np.concatenate(feats)
    #endregion


def features_of(matrices, photos_tr, photos_te):
    """
    * what? - Loads the original body with the 72 matrices replaced and returns CLS features of train and test.
    * why?  - Every variant is one set of 72 matrices; the rest of the body is always the original.
    * how?  - Dinov2Model from the original folder, state dict patched, extract_cached on both photo tensors.
    """
    #region 1 - Patch the body and extract
    # what? - Two feature arrays (N x 384).
    # how?  - The dtype of the original entries is kept (float32).
    body = Dinov2Model.from_pretrained(str(ORIGINAL))
    sd = body.state_dict()
    for n, W in matrices.items():
        sd[n] = W.to(sd[n].dtype)
    body.load_state_dict(sd)
    body = body.to(DEVICE).eval()
    x_tr, x_te = extract_cached(body, photos_tr), extract_cached(body, photos_te)
    del body
    torch.cuda.empty_cache()
    return x_tr, x_te
    #endregion


def scores(x_tr, y_tr, x_te, y_te):
    """
    * what? - Returns validation and test accuracy of a logistic regression on the features.
    * why?  - Validation (a held-out tenth of the train photos) chooses k; test reports the result.
    * how?  - Split with a fixed seed; fit on 90% (up to 1,000 iterations); score on the 10%; then refit on all train with the screening protocol (3,000 iterations) for the test number.
    """
    #region 1 - Validation split
    # what? - Accuracy on the held-out tenth.
    # how?  - Same seed and fraction as the stage 4 pilots.
    g = np.random.default_rng(VALIDATION_SEED)
    idx = g.permutation(len(x_tr))
    n_val = int(round(VALIDATION_FRACTION * len(x_tr)))
    val, fit = idx[:n_val], idx[n_val:]
    clf = LogisticRegression(max_iter=1000, C=1.0).fit(x_tr[fit], y_tr[fit])   # the validation fit only chooses k
    val_acc = 100 * clf.score(x_tr[val], y_tr[val])
    #endregion

    #region 2 - Test with all train photos
    # what? - Accuracy on the 2,840 test photos.
    # how?  - Same protocol as domain_screening.py.
    clf = LogisticRegression(max_iter=3000, C=1.0).fit(x_tr, y_tr)
    return round(val_acc, 2), round(100 * clf.score(x_te, y_te), 2)
    #endregion


def main():
    """
    * what? - Runs the base, the predicted adapters, the random controls and the learned delta for the 10 seeds, and writes the JSON.
    * why?  - Gives session 17 its numbers.
    * how?  - One feature extraction per variant and seed; logistic regression on the features.
    """
    #region 1 - Data, transform, original matrices and their SVD
    # what? - Tables, labels, the 72 original matrices and U0, S0, Vt0 of each.
    # how?  - Transform copied from domain_screening.py (resize 256, center crop 224, ImageNet normalization).
    warnings.filterwarnings("ignore", category=ConvergenceWarning)   # the 3,000-iteration limit is the screening protocol; the warning is noted in the diary
    tr, te = skin_tables()
    y_tr, y_te = tr["label"].values, te["label"].values
    transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    photos_tr, photos_te = cache_photos(transform, tr), cache_photos(transform, te)
    print(f"photos cached: train {tuple(photos_tr.shape)} test {tuple(photos_te.shape)}", flush=True)
    original = load_file(str(ORIGINAL / "model.safetensors"))
    names = [n for n, t in original.items() if t.dim() == 2 and "encoder.layer." in n]
    svd = {n: torch.linalg.svd(original[n].float(), full_matrices=False) for n in names}
    body = Dinov2Model.from_pretrained(str(ORIGINAL))
    names_in_param_order = [n for n, p in body.named_parameters() if p.dim() == 2 and n.startswith("encoder.layer.")]
    del body
    #endregion

    #region 2 - Base
    # what? - The frozen original body, the triage number.
    # how?  - No matrix replaced.
    x_tr, x_te = features_of({}, photos_tr, photos_te)
    base_val, base_test = scores(x_tr, y_tr, x_te, y_te)
    print(f"base: validation {base_val} test {base_test}", flush=True)
    #endregion

    #region 3 - Per seed: predicted, random and learned
    # what? - Validation and test accuracy of every variant.
    # how?  - build(delta) rebuilds the 72 matrices from a dict of per-matrix deltas; the random delta matches the norm of the predicted one, matrix by matrix.
    def build(delta):
        """
        * what? - Rebuilds the 72 matrices from a dict of per-matrix deltas on the original directions.
        * why?  - Predicted, random and learned deltas all become matrices through this one formula.
        * how?  - W = U0 diag(S0 (1 + delta)) V0^T, with the SVD factors computed once above.
        """
        #region 1 - Rebuild
        # what? - Dict name -> matrix.
        # how?  - Column scaling of U0, then the product with Vt0.
        return {n: (svd[n][0] * (svd[n][1] * (1 + delta[n]))) @ svd[n][2] for n in names}
        #endregion
    rows = []
    for s in SEEDS:
        free = load_file(str(ROOT / f"models/etapa3/s{s}/model.safetensors"))
        learned_file = torch.load(ROOT / f"models/etapa5/so_sigma_lr3e-2_s{s}/parametros_restritos.pt", map_location="cpu")
        foot = {}
        for n in names:
            U0, S0, Vt0 = svd[n]
            dW = free[n].float() - original[n].float()
            foot[n] = ((U0.T @ dW) * Vt0).sum(1) / S0
        row = {"seed": s, "footprint_mean_abs_pct": 100 * st.mean(f.abs().mean().item() for f in foot.values()), "previsto": {}, "aleatorio": {}}
        gen = torch.Generator().manual_seed(7000 + s)
        for k in KS:
            pred = {n: k * foot[n] for n in names}
            rand = {}
            for n in names:
                r = torch.randn(pred[n].shape, generator=gen)
                rand[n] = r * (pred[n].norm() / r.norm())
            x_tr, x_te = features_of(build(pred), photos_tr, photos_te)
            row["previsto"][str(k)] = dict(zip(("validacao", "teste"), scores(x_tr, y_tr, x_te, y_te)))
            x_tr, x_te = features_of(build(rand), photos_tr, photos_te)
            row["aleatorio"][str(k)] = dict(zip(("validacao", "teste"), scores(x_tr, y_tr, x_te, y_te)))
        learned = {n: learned_file[f"restrito_{i}"].float() for i, n in enumerate(names_in_param_order)}
        row["aprendido_mean_abs_pct"] = 100 * st.mean(v.abs().mean().item() for v in learned.values())
        x_tr, x_te = features_of(build(learned), photos_tr, photos_te)
        row["aprendido"] = dict(zip(("validacao", "teste"), scores(x_tr, y_tr, x_te, y_te)))
        rows.append(row)
        print(json.dumps(row), flush=True)
    #endregion

    #region 4 - Choose k by validation and summarize
    # what? - The k with the best mean validation accuracy of the predicted adapter, and test means at that k.
    # how?  - Means over the 10 seeds; the random control is read at the same k.
    best_k = max(KS, key=lambda k: st.mean(r["previsto"][str(k)]["validacao"] for r in rows))
    summary = {"base_test": base_test, "k_escolhido": best_k,
               "previsto_teste": {str(k): st.mean(r["previsto"][str(k)]["teste"] for r in rows) for k in KS},
               "aleatorio_teste": {str(k): st.mean(r["aleatorio"][str(k)]["teste"] for r in rows) for k in KS},
               "previsto_validacao": {str(k): st.mean(r["previsto"][str(k)]["validacao"] for r in rows) for k in KS},
               "aprendido_teste": st.mean(r["aprendido"]["teste"] for r in rows),
               "footprint_mean_abs_pct": st.mean(r["footprint_mean_abs_pct"] for r in rows),
               "aprendido_mean_abs_pct": st.mean(r["aprendido_mean_abs_pct"] for r in rows)}
    OUTPUT.write_text(json.dumps({"ks": KS, "rows": rows, "summary": summary}, indent=1))
    print("summary:", json.dumps(summary))
    print("written:", OUTPUT.relative_to(ROOT))
    #endregion


if __name__ == "__main__":
    main()
