"""
* what? - Diagnosis of attempt 0: does the FROZEN DINOv2 body already separate the 37 breeds?
* why?  - If a frozen body scores high, the problem of attempt 0 was the training method, not the body.
* how?  - Extracts the CLS token (384 numbers) per photo with the frozen body and fits only a logistic regression.

Rewritten in stage 2: the original script was lost when the temporary directory was cleaned.
The original value recorded in stage 1 was 94.8%.
Run from the repository root:  python3 doutorado/experimento1/etapa1/scripts/diagnose_frozen_body.py
"""
import io
import json
import time

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from torchvision import transforms
from transformers import Dinov2Model

DEVICE = "cuda"


def extract(model, transform, path):
    """
    * what? - Returns the CLS features (N x 384) and the breed labels of every photo in a parquet file.
    * why?  - The logistic regression needs the frozen-body features as a plain array.
    * how?  - Batches of 64 photos go through the body under bf16 autocast; token 0 is the CLS.
    """
    #region 1 - Read the table
    # what? - Loads the parquet and prepares the accumulators.
    # how?  - One list entry per batch.
    df = pd.read_parquet(path)
    feats, labels = [], []
    #endregion

    #region 2 - Extract the features batch by batch
    # what? - Runs the frozen body on every batch.
    # how?  - no_grad plus autocast, same numerics as training; features are moved to CPU as float32.
    for i in range(0, len(df), 64):
        chunk = df.iloc[i:i + 64]
        photos = torch.stack([transform(Image.open(io.BytesIO(r["image"]["bytes"])).convert("RGB"))
                              for _, r in chunk.iterrows()]).to(DEVICE)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            cls = model(pixel_values=photos).last_hidden_state[:, 0]
        feats.append(cls.float().cpu().numpy())
        labels.append(chunk["label"].values)
    return np.concatenate(feats), np.concatenate(labels)
    #endregion


def main():
    """
    * what? - Fits a logistic regression on frozen-body features and writes diagnostico.json.
    * why?  - Gives the frozen-body accuracy that stage 1 used to judge attempt 0.
    * how?  - Extract train and test features, fit LogisticRegression(C=1.0), score on test.
    """
    #region 1 - Load the frozen body and the transform
    # what? - Builds the evaluation preprocessing and the eval-mode body.
    # how?  - Resize 256, center crop 224 and ImageNet normalization, as in the test pipeline of train.py.
    transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    model = Dinov2Model.from_pretrained("models/dinov2-small").to(DEVICE).eval()
    #endregion

    #region 2 - Extract features and fit the classifier
    # what? - Trains only the logistic head on top of frozen features.
    # how?  - max_iter 2000 to make sure the solver converges.
    start = time.time()
    x_train, y_train = extract(model, transform, "data/hf/data/train-00000-of-00001.parquet")
    x_test, y_test = extract(model, transform, "data/hf/data/test-00000-of-00001.parquet")
    clf = LogisticRegression(max_iter=2000, C=1.0).fit(x_train, y_train)
    acc = 100 * clf.score(x_test, y_test)
    print(f"ACCURACY with the FROZEN body (only the head trained): {acc:.2f}%  ({time.time() - start:.0f}s)")
    #endregion

    #region 3 - Write the result
    # what? - Saves the accuracy next to the value recorded in stage 1.
    # how?  - The keys stay in Portuguese: later scripts read this schema.
    json.dump({"acuracia_corpo_congelado": round(acc, 2), "valor_original_etapa1": 94.8,
               "nota": "script reescrito; o original foi perdido"},
              open("doutorado/experimento1/etapa1/diagnostico.json", "w"), indent=2)
    #endregion


if __name__ == "__main__":
    main()
