"""
* what? - STAGE 3, step 1: screens the candidate domains with the FROZEN DINOv2-Small and a logistic regression.
* why?  - The lower the frozen-body accuracy, the more the body must change in fine-tuning: that picks the domain.
* how?  - Extracts the CLS token with the same sub-sample and test transforms the training will use; fits LogisticRegression.

Output: etapa3/triagem.json
Run from the repository root:  python3 doutorado/experimento1/etapa3/scripts/domain_screening.py
"""
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from torchvision import transforms
from transformers import Dinov2Model

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config_etapa3 as C  # noqa: E402

DEVICE = "cuda"
# the "nome" strings are page content and stay in Portuguese
CANDIDATES = {
    "dtd":     {"nome": "DTD (texturas)",            "treino": "dtd/data/train-00000-of-00001.parquet",     "teste": "dtd/data/test-00000-of-00001.parquet"},
    "eurosat": {"nome": "EuroSAT RGB (satelite)",    "treino": "eurosat/data/train-00000-of-00001.parquet", "teste": "eurosat/data/test-00000-of-00001.parquet"},
    "pele":    {"nome": "Lesoes de pele (8 classes)", "treino": "pele/data/train-00000-of-00001-2b84ce2361bc8c42.parquet", "teste": "pele/data/test-00000-of-00001-3a9b253c406a4af7.parquet"},
    "kmnist":  {"nome": "KMNIST (caracteres japoneses manuscritos)", "treino": "kmnist/kmnist/train-00000-of-00001.parquet", "teste": "kmnist/kmnist/test-00000-of-00001.parquet"},
}


def open_image(value):
    """
    * what? - Opens an image cell of a parquet table as an RGB PIL image.
    * why?  - Cells are either {'bytes': ...} dicts or raw bytes depending on the dataset.
    * how?  - Picks the bytes and decodes them in memory.
    """
    #region 1 - Decode the image
    # what? - Returns the RGB image.
    # how?  - io.BytesIO avoids touching the disk.
    return Image.open(io.BytesIO(value["bytes"] if isinstance(value, dict) else value)).convert("RGB")
    #endregion


def extract(model, transform, df):
    """
    * what? - Returns the CLS features (N x 384) of every photo of a table, from the frozen body.
    * why?  - The logistic regression needs a plain feature array.
    * how?  - Batches of 64 under bf16 autocast; token 0 of the last hidden state is the CLS.
    """
    #region 1 - Extract batch by batch
    # what? - Runs the frozen body and collects the features on CPU.
    # how?  - no_grad plus autocast, same numerics as training.
    feats = []
    images = df["image"].tolist()
    for i in range(0, len(images), 64):
        x = torch.stack([transform(open_image(v)) for v in images[i:i + 64]]).to(DEVICE)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            feats.append(model(pixel_values=x).last_hidden_state[:, 0].float().cpu().numpy())
    return np.concatenate(feats)
    #endregion


def main():
    """
    * what? - Screens the four candidates and writes triagem.json.
    * why?  - The screening table is the evidence behind choosing skin lesions as the stage 3 domain.
    * how?  - Same sub-sampling order as train_etapa3.py; frozen features, logistic regression, accuracies and baselines.
    """
    #region 1 - Load the frozen body
    # what? - Evaluation transform and eval-mode model.
    # how?  - Resize 256, center crop 224 and ImageNet normalization.
    transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    model = Dinov2Model.from_pretrained(str(C.ORIGINAL_MODEL)).to(DEVICE).eval()
    #endregion

    #region 2 - Screen every candidate
    # what? - Features, logistic regression and baselines per candidate.
    # how?  - The generator draws train first and test second, like train_etapa3.py, so the sub-samples match.
    output = {}
    for nickname, c in CANDIDATES.items():
        t = time.time()
        tr, te = pd.read_parquet(C.DATA / c["treino"]), pd.read_parquet(C.DATA / c["teste"])
        g = np.random.default_rng(C.SUBSAMPLE_SEED)
        if len(tr) > C.MAX_TRAIN:
            tr = tr.iloc[np.sort(g.choice(len(tr), C.MAX_TRAIN, replace=False))]
        if len(te) > C.MAX_TEST:
            te = te.iloc[np.sort(g.choice(len(te), C.MAX_TEST, replace=False))]
        x_train, x_test = extract(model, transform, tr), extract(model, transform, te)
        y_train, y_test = tr["label"].values, te["label"].values
        clf = LogisticRegression(max_iter=3000, C=1.0).fit(x_train, y_train)
        k = len(np.unique(y_train))
        majority = pd.Series(y_test).value_counts(normalize=True).max()
        output[nickname] = {"nome": c["nome"], "treino_parquet": c["treino"], "teste_parquet": c["teste"],
                            "n_treino": int(len(tr)), "n_teste": int(len(te)), "classes": int(k),
                            "acuracia_corpo_congelado": round(100 * clf.score(x_test, y_test), 2), "acuracia_treino": round(100 * clf.score(x_train, y_train), 2),
                            "acaso_pct": round(100 / k, 2), "classe_majoritaria_pct": round(100 * majority, 2), "segundos": round(time.time() - t)}
        print(nickname, json.dumps(output[nickname], ensure_ascii=False), flush=True)
    #endregion

    #region 3 - Add the reference and write the file
    # what? - Adds the Oxford-IIIT Pet reference row and saves triagem.json.
    # how?  - The reference value 94.79 comes from etapa1/diagnostico.json.
    output["pets_referencia"] = {"nome": "Oxford-IIIT Pet (etapas 1 e 2)", "acuracia_corpo_congelado": 94.79, "classes": 37, "acaso_pct": 2.7,
                                 "fonte": "etapa1/diagnostico.json"}
    json.dump(output, open(C.STAGE3 / "triagem.json", "w"), indent=2, ensure_ascii=False)
    print("written to etapa3/triagem.json")
    #endregion


if __name__ == "__main__":
    main()
