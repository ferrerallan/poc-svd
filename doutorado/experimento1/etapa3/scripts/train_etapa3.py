"""
* what? - STAGE 3: fine-tunes DINOv2-Small on a DISTANT domain with the SAME protocol as stage 2.
* why?  - Stage 1-2 used a task the model already solves; a distant domain forces the body to really change.
* how?  - Adapted copy of common/train.py (left untouched) with a generic parquet dataset and a fixed sub-sample.

Differences from common/train.py:
  - generic parquet dataset (--train/--test, --image-col/--label-col), labels mapped to 0..K-1
  - fixed sub-sample (SUBSAMPLE_SEED) of up to 3680 train and 3669 test photos, the same for every seed
  - weight_decay = 0 by default (stage 2)
  - --shuffle-labels: control with permuted train labels (same permutation in every seed)
  - relatorio.json with train accuracy per epoch, class map and metadata
Everything else is identical: AdamW, step 1e-5 (body) and 1e-3 (head), 5 epochs, batch 32, cosine, label smoothing 0.1,
bf16, body frozen in epoch 1, new linear head on the CLS token.
"""
import argparse
import datetime
import hashlib
import io
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import transformers
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from transformers import Dinov2Model

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config_etapa3 as C  # noqa: E402

EPOCHS, WARMUP_EPOCHS, BATCH_SIZE, BODY_LR, HEAD_LR = 5, 1, 32, 1e-5, 1e-3


class Classifier(nn.Module):
    """
    * what? - DINOv2 body with a new linear head on top of the CLS token.
    * why?  - Same architecture as stage 1-2, with the number of classes of the new domain.
    * how?  - Dinov2Model.from_pretrained plus nn.Linear(hidden_size, num_classes).
    """

    def __init__(self, model_path, num_classes):
        """
        * what? - Builds the body from a folder and the head.
        * why?  - The head starts random; only the body comes pre-trained.
        * how?  - Order matters for reproducibility: body first, then the head draws its random numbers.
        """
        #region 1 - Build body and head
        # what? - Creates the two sub-modules.
        # how?  - hidden_size comes from the body config.
        super().__init__()
        self.body = Dinov2Model.from_pretrained(model_path)
        self.head = nn.Linear(self.body.config.hidden_size, num_classes)
        #endregion

    def forward(self, pixels):
        """
        * what? - Maps a batch of photos to one score per class.
        * why?  - Classification reads only the CLS token.
        * how?  - Body, token 0 of the last hidden state, head.
        """
        #region 1 - Run body and head
        # what? - Returns the logits.
        # how?  - last_hidden_state[:, 0] is the CLS token.
        return self.head(self.body(pixel_values=pixels).last_hidden_state[:, 0])
        #endregion


def open_image(value):
    """
    * what? - Opens an image cell of a parquet table as an RGB PIL image.
    * why?  - Some datasets store {'bytes': ...} dicts and others raw bytes.
    * how?  - Picks the bytes from either form and decodes them in memory.
    """
    #region 1 - Decode the image
    # what? - Returns the RGB image.
    # how?  - io.BytesIO avoids touching the disk.
    b = value["bytes"] if isinstance(value, dict) else value
    return Image.open(io.BytesIO(b)).convert("RGB")
    #endregion


class ParquetImages(Dataset):
    """
    * what? - Dataset over a list of image cells and an array of integer labels.
    * why?  - Stage 3 reads arbitrary parquet datasets whose labels were already mapped to 0..K-1.
    * how?  - Keeps the cells and labels in memory; decodes lazily in __getitem__.
    """

    def __init__(self, df, image_col, labels, transform):
        """
        * what? - Stores the images, labels and transform.
        * why?  - Decoding is lazy to keep memory low.
        * how?  - tolist() of the image column.
        """
        #region 1 - Store the data
        # what? - Keeps references.
        # how?  - Attribute names: images, labels, transform.
        self.images, self.labels, self.transform = df[image_col].tolist(), labels, transform
        #endregion

    def __len__(self):
        """
        * what? - Returns the number of items.
        * why?  - DataLoader needs the length.
        * how?  - Length of the labels array.
        """
        #region 1 - Return the length
        # what? - Counts the labels.
        # how?  - len().
        return len(self.labels)
        #endregion

    def __getitem__(self, i):
        """
        * what? - Returns the transformed photo and the integer label of item i.
        * why?  - The training loop consumes (photo, label) pairs.
        * how?  - Decodes the image and applies the transform.
        """
        #region 1 - Build the item
        # what? - (tensor, int).
        # how?  - The label is cast to int for the loss.
        return self.transform(open_image(self.images[i])), int(self.labels[i])
        #endregion


def build_data(a):
    """
    * what? - Builds train and test datasets of a domain with a fixed sub-sample, label mapping and optional shuffling.
    * why?  - Every seed must see the same photos, and the control needs the same permutation in every seed.
    * how?  - Reads the parquet files, maps labels to 0..K-1, sub-samples with SUBSAMPLE_SEED, permutes with LABEL_SEED.
    """
    #region 1 - Define the transforms
    # what? - Train and test preprocessing, as in common/train.py.
    # how?  - ImageNet normalization; train adds random crop and flip.
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    train_tf = transforms.Compose([transforms.RandomResizedCrop(224, scale=(0.7, 1.0)), transforms.RandomHorizontalFlip(),
                                   transforms.ToTensor(), transforms.Normalize(mean, std)])
    test_tf = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(mean, std)])
    #endregion

    #region 2 - Read the tables and map the labels
    # what? - Loads the parquet files and builds the class map.
    # how?  - Classes sorted by (type name, value) so mixed label types stay deterministic.
    tr, te = pd.read_parquet(a.train), pd.read_parquet(a.test)
    classes = sorted(tr[a.label_col].unique().tolist(), key=lambda x: (str(type(x)), x))
    mapping = {c: i for i, c in enumerate(classes)}
    #endregion

    #region 3 - Sub-sample with the fixed seed
    # what? - Keeps at most max_train / max_test photos, the same for every training seed.
    # how?  - The generator draws train first and test second; the order matters and is kept from the original.
    g = np.random.default_rng(C.SUBSAMPLE_SEED)
    if len(tr) > a.max_train:
        tr = tr.iloc[np.sort(g.choice(len(tr), a.max_train, replace=False))]
    if len(te) > a.max_test:
        te = te.iloc[np.sort(g.choice(len(te), a.max_test, replace=False))]
    ytr = np.array([mapping[c] for c in tr[a.label_col]])
    yte = np.array([mapping.get(c, -1) for c in te[a.label_col]])
    assert (yte >= 0).all(), "test class that does not exist in train"
    #endregion

    #region 4 - Shuffle the train labels for the control
    # what? - Permutes the train labels when asked.
    # how?  - Fixed LABEL_SEED, so the permutation is identical in every seed.
    if a.shuffle_labels:
        ytr = np.random.default_rng(C.LABEL_SEED).permutation(ytr)
    return (ParquetImages(tr, a.image_col, ytr, train_tf), ParquetImages(te, a.image_col, yte, test_tf),
            [str(c) for c in classes])
    #endregion


@torch.no_grad()
def accuracy(model, loader, device):
    """
    * what? - Returns the percentage of correct predictions of a model over a loader.
    * why?  - Shared by the stage 3 and stage 4 trainings and by the transplant test.
    * how?  - Eval mode, bf16 autocast, argmax over the class scores.
    """
    #region 1 - Count the hits
    # what? - Accumulates hits and total.
    # how?  - Same autocast as training.
    model.eval()
    hits = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            hits += (model(x).argmax(1) == y).sum().item()
            total += y.numel()
    #endregion

    #region 2 - Return the percentage
    # what? - Converts counts to a percentage.
    # how?  - 100 * hits / total.
    return 100 * hits / total
    #endregion


def main():
    """
    * what? - Trains one seed on one dataset and writes the model, the head and relatorio.json.
    * why?  - Produces the before/after pair of stage 3 for the domain pilots, the 10 seeds and the shuffled control.
    * how?  - Parse flags, build data and model, train with the frozen-body warm-up, save and report.
    """
    #region 1 - Parse flags and seed
    # what? - Reads the command line and fixes the seed.
    # how?  - torch.manual_seed before the model is built.
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--image-col", default="image", dest="image_col")
    ap.add_argument("--label-col", default="label", dest="label_col")
    ap.add_argument("--dataset", required=True, help="dataset nickname, only for the report")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", required=True)
    ap.add_argument("--weight-decay", type=float, default=C.WEIGHT_DECAY, dest="weight_decay")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--warmup-epochs", type=int, default=WARMUP_EPOCHS, dest="warmup_epochs")
    ap.add_argument("--body-lr", type=float, default=BODY_LR, dest="body_lr")
    ap.add_argument("--max-train", type=int, default=C.MAX_TRAIN, dest="max_train")
    ap.add_argument("--max-test", type=int, default=C.MAX_TEST, dest="max_test")
    ap.add_argument("--shuffle-labels", action="store_true", dest="shuffle_labels")
    a = ap.parse_args()
    started_iso = datetime.datetime.now().isoformat(timespec="seconds")
    torch.manual_seed(a.seed)
    device = "cuda"
    #endregion

    #region 2 - Prepare the data
    # what? - Builds datasets and loaders.
    # how?  - Train shuffles and drops the last partial batch.
    train, test, classes = build_data(a)
    print(f"1) data: {a.dataset} | train {len(train)} | test {len(test)} | classes {len(classes)} | shuffled labels: {a.shuffle_labels}")
    train_loader = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
    #endregion

    #region 3 - Build the model and the optimizer
    # what? - Classifier, optimizer, schedule and loss.
    # how?  - Body and head have their own learning rates.
    model = Classifier(str(C.ORIGINAL_MODEL), len(classes)).to(device)
    optimizer = torch.optim.AdamW([{"params": model.body.parameters(), "lr": a.body_lr},
                                   {"params": model.head.parameters(), "lr": HEAD_LR}], weight_decay=a.weight_decay)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, a.epochs * len(train_loader))
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    acc0 = accuracy(model, test_loader, device)
    print(f"   accuracy before training: {acc0:.1f}%")
    #endregion

    #region 4 - Train
    # what? - Epoch loop with the frozen-body warm-up; records test and train accuracy and loss per epoch.
    # how?  - The body unfreezes after the warm-up epochs.
    start = time.time()
    acc_epoch, train_acc_epoch, loss_epoch = [], [], []
    for epoch in range(1, a.epochs + 1):
        free = epoch > a.warmup_epochs
        for p in model.body.parameters():
            p.requires_grad_(free)
        model.train()
        loss_sum = 0.0
        hits = total = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = model(x)
                loss = loss_fn(output, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            schedule.step()
            loss_sum += loss.item()
            hits += (output.argmax(1) == y).sum().item()
            total += y.numel()
        acc = accuracy(model, test_loader, device)
        acc_epoch.append(round(acc, 2))
        train_acc_epoch.append(round(100 * hits / total, 2))
        loss_epoch.append(round(loss_sum / len(train_loader), 4))
        print(f"   epoch {epoch}/{a.epochs}  {'body free  ' if free else 'body frozen'}  loss {loss_sum/len(train_loader):.3f}  train acc {100*hits/total:.1f}%  test acc {acc:.1f}%")
    minutes = (time.time() - start) / 60
    final_acc = accuracy(model, test_loader, device)
    #endregion

    #region 5 - Save the model and the report
    # what? - Writes the body, the head and relatorio.json.
    # how?  - The report keys stay in Portuguese: later scripts and the page builders read this schema.
    dest = Path(a.output)
    dest.mkdir(parents=True, exist_ok=True)
    model.body.save_pretrained(dest)
    torch.save(model.head.state_dict(), dest / "cabeca.pt")
    report = {"etapa": 3, "dataset": a.dataset, "treino_parquet": a.train, "teste_parquet": a.test,
              "col_imagem": a.image_col, "col_rotulo": a.label_col, "num_classes": len(classes), "classes": classes,
              "n_treino": len(train), "n_teste": len(test), "max_treino": a.max_train, "max_teste": a.max_test,
              "semente_subamostra": C.SUBSAMPLE_SEED, "rotulos_embaralhados": a.shuffle_labels,
              "semente_rotulos": C.LABEL_SEED if a.shuffle_labels else None,
              "modelo_antes": str(C.ORIGINAL_MODEL), "modelo_depois": a.output, "semente": a.seed,
              "epocas": a.epochs, "epocas_aquecer": a.warmup_epochs, "lote": BATCH_SIZE, "lr_corpo": a.body_lr, "lr_cabeca": HEAD_LR,
              "weight_decay": a.weight_decay, "label_smoothing": 0.1, "otimizador": "AdamW", "agenda_lr": "cosseno, um passo por lote",
              "precisao": "bf16 (autocast)", "acuracia_antes": round(acc0, 2), "acuracia_depois": round(final_acc, 2),
              "acuracias_por_epoca": acc_epoch, "acuracia_treino_por_epoca": train_acc_epoch, "perda_por_epoca": loss_epoch,
              "minutos_de_treino": round(minutes, 1), "comando": " ".join(sys.argv), "iniciado_em": started_iso,
              "python": platform.python_version(), "torch": torch.__version__, "transformers": transformers.__version__,
              "gpu": torch.cuda.get_device_name(0), "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (dest / "relatorio.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nDONE in {minutes:.1f} min | accuracy: {acc0:.1f}% -> {final_acc:.1f}% | saved in {a.output}/")
    #endregion


if __name__ == "__main__":
    main()
