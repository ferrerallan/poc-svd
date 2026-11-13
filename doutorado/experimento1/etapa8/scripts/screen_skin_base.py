"""
* what? - STAGE 8 (session 19): the frozen-body triage of session 12 (CLS features plus logistic regression) applied to skin lesions with DINOv2-Base.
* why?  - Before training, session 19 needs the first number of the larger model: how far the frozen Base already is from the task, next to the 56.58% of the Small.
* how?  - Reuses extract() and open_image() of etapa3/scripts/domain_screening.py with the same sub-sample and test transforms, pointing the model at models/dinov2-base; writes etapa8/triagem_base.json and never touches etapa3/triagem.json.

Run from the repository root:  python3 doutorado/experimento1/etapa8/scripts/screen_skin_base.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from torchvision import transforms
from transformers import Dinov2Model

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "etapa3" / "scripts"))
import config_etapa3 as C3  # noqa: E402
import domain_screening as DS  # noqa: E402

ROOT = HERE.parents[3]
BASE_MODEL = ROOT / "models/dinov2-base"
OUTPUT = ROOT / "doutorado/experimento1/etapa8/triagem_base.json"
SMALL_TRIAGE = ROOT / "doutorado/experimento1/etapa3/triagem.json"


def main():
    """
    * what? - Screens skin lesions with the frozen DINOv2-Base and writes triagem_base.json.
    * why?  - Gives session 19 its frozen-body reference for the larger model.
    * how?  - Same sub-sampling, transforms and classifier as domain_screening.py; only the model path and the output file differ.
    """
    #region 1 - Load the frozen body and the skin tables
    # what? - Evaluation transform, eval-mode Base model and the same sub-sample as every skin training.
    # how?  - The generator draws train first and test second, like train_etapa3.py and domain_screening.py.
    transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    model = Dinov2Model.from_pretrained(str(BASE_MODEL)).to(DS.DEVICE).eval()
    skin = DS.CANDIDATES["pele"]
    tr, te = pd.read_parquet(C3.DATA / skin["treino"]), pd.read_parquet(C3.DATA / skin["teste"])
    g = np.random.default_rng(C3.SUBSAMPLE_SEED)
    if len(tr) > C3.MAX_TRAIN:
        tr = tr.iloc[np.sort(g.choice(len(tr), C3.MAX_TRAIN, replace=False))]
    if len(te) > C3.MAX_TEST:
        te = te.iloc[np.sort(g.choice(len(te), C3.MAX_TEST, replace=False))]
    #endregion

    #region 2 - Features, logistic regression and the comparison with the Small
    # what? - Frozen accuracy of the Base on the test set, next to the Small value of etapa3/triagem.json.
    # how?  - LogisticRegression(max_iter=3000, C=1.0), the same instrument as session 12.
    t = time.time()
    x_train, x_test = DS.extract(model, transform, tr), DS.extract(model, transform, te)
    y_train, y_test = tr["label"].values, te["label"].values
    clf = LogisticRegression(max_iter=3000, C=1.0).fit(x_train, y_train)
    small = json.load(open(SMALL_TRIAGE))["pele"]
    out = {"modelo": "dinov2-base", "largura_cls": int(x_train.shape[1]), "n_treino": int(len(tr)), "n_teste": int(len(te)), "classes": int(len(np.unique(y_train))),
           "acuracia_corpo_congelado": round(100 * clf.score(x_test, y_test), 2), "acuracia_treino": round(100 * clf.score(x_train, y_train), 2),
           "acaso_pct": small["acaso_pct"], "segundos": round(time.time() - t),
           "small_referencia": {"acuracia_corpo_congelado": small["acuracia_corpo_congelado"], "acuracia_treino": small["acuracia_treino"], "fonte": "etapa3/triagem.json"}}
    OUTPUT.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps(out, ensure_ascii=False))
    print("written: doutorado/experimento1/etapa8/triagem_base.json")
    #endregion


if __name__ == "__main__":
    main()
