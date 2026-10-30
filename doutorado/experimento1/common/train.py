"""
* what? - Experiment 1 training: fine-tunes the ready-made DINOv2-Small on 37 pet breeds (Oxford-IIIT Pet).
* why?  - The goal is not the breed classifier; it is the before/after pair of weights that SVD will compare.
* how?  - AdamW with a frozen-body warm-up; defaults are stage 1 (weight_decay 0.05), stage 2 only passes --weight-decay 0.

Run from the repository root:
    python3 doutorado/experimento1/common/train.py --seed 0 --output models/dinov2-small-pets
BEFORE -> models/dinov2-small/        (already exists, never touched)
AFTER  -> --output folder             (created by this script)
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

import pandas as pd
import torch
import torch.nn as nn
import transformers
from huggingface_hub import hf_hub_download
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from transformers import Dinov2Model

# ------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------
MODEL_BEFORE = "models/dinov2-small"
MODEL_AFTER = "models/dinov2-small-pets"   # can be replaced with --output
DATA_DIR = "data"

NUM_CLASSES = 37       # the 37 breeds
EPOCHS = 5
WARMUP_EPOCHS = 1      # during the first epoch the body stays frozen
BATCH_SIZE = 32
BODY_LR = 1e-5         # very small step: the body already knows a lot
HEAD_LR = 1e-3         # large step: the head starts from zero
SEED = 0
# ------------------------------------------------------------------


class PetClassifier(nn.Module):
    """
    * what? - DINOv2 body with a 37-output linear head glued on top of the CLS token.
    * why?  - The body alone has no head; the head is what lets a breed label pull on the body weights.
    * how?  - Dinov2Model.from_pretrained for the body and one nn.Linear(hidden_size, num_classes) as head.
    """

    def __init__(self, model_path, num_classes=37):
        """
        * what? - Builds the body from a folder and creates the head.
        * why?  - Only the body is saved in the original format; the head state is saved separately as cabeca.pt.
        * how?  - Loads the pretrained body, then a linear layer sized by the body hidden size.
        """
        #region 1 - Build body and head
        # what? - Creates the two sub-modules.
        # how?  - The head starts with random weights, which is why it gets the larger learning rate.
        super().__init__()
        self.body = Dinov2Model.from_pretrained(model_path)
        self.head = nn.Linear(self.body.config.hidden_size, num_classes)
        #endregion

    def forward(self, pixels):
        """
        * what? - Maps a batch of photos to one score per breed.
        * why?  - Classification reads only the CLS token, the summary of the photo.
        * how?  - Runs the body, keeps token 0 of the last hidden state and applies the head to it.
        """
        #region 1 - Run body and head
        # what? - Produces the logits.
        # how?  - last_hidden_state[:, 0] is the CLS token of every photo in the batch.
        output = self.body(pixel_values=pixels).last_hidden_state
        cls_token = output[:, 0]
        return self.head(cls_token)
        #endregion


class PetParquet(Dataset):
    """
    * what? - Dataset that reads photos and breed labels straight from a HuggingFace parquet file.
    * why?  - The original Oxford server serves 70 KB/s, so we use the HuggingFace mirror of the same dataset.
    * how?  - Loads the table once; each item decodes the JPEG bytes and applies the given transform.
    """

    def __init__(self, parquet_path, transform):
        """
        * what? - Loads the parquet table and stores the transform.
        * why?  - Decoding happens lazily in __getitem__ to keep memory low.
        * how?  - pandas.read_parquet keeps the image bytes inside the table.
        """
        #region 1 - Load the table
        # what? - Reads the whole file into memory.
        # how?  - The image column holds dicts with the raw JPEG bytes.
        self.table = pd.read_parquet(parquet_path)
        self.transform = transform
        #endregion

    def __len__(self):
        """
        * what? - Returns the number of photos.
        * why?  - DataLoader needs the dataset length.
        * how?  - Length of the parquet table.
        """
        #region 1 - Return the length
        # what? - Counts rows.
        # how?  - len() of the table.
        return len(self.table)
        #endregion

    def __getitem__(self, i):
        """
        * what? - Returns the transformed photo tensor and the breed number of item i.
        * why?  - The training loop consumes (photo, label) pairs.
        * how?  - Decodes the JPEG bytes to RGB, applies the transform and casts the label to int.
        """
        #region 1 - Decode one item
        # what? - Builds the (tensor, label) pair.
        # how?  - PIL decodes from an in-memory buffer; no file is written.
        row = self.table.iloc[i]
        photo = Image.open(io.BytesIO(row["image"]["bytes"])).convert("RGB")
        return self.transform(photo), int(row["label"])
        #endregion


def build_data():
    """
    * what? - Downloads (if missing) the Oxford-IIIT Pet parquet files and builds train and test datasets.
    * why?  - Train and test need different preprocessing: random crops to learn, fixed center crop to measure.
    * how?  - hf_hub_download into data/hf, then PetParquet with the matching torchvision transform.
    """
    #region 1 - Define the transforms
    # what? - Normalization constants and the two preprocessing pipelines.
    # how?  - ImageNet mean and std; train adds a random crop and a horizontal flip.
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    #endregion

    #region 2 - Fetch the parquet files
    # what? - Makes sure both parquet files exist locally.
    # how?  - hf_hub_download is a no-op when the file is already in local_dir.
    files = {}
    for name, part in [("train", "train"), ("test", "test")]:
        files[name] = hf_hub_download(
            "timm/oxford-iiit-pet",
            f"data/{part}-00000-of-00001.parquet",
            repo_type="dataset",
            local_dir=DATA_DIR + "/hf",
        )
    #endregion

    #region 3 - Build the datasets
    # what? - Wraps each file in a PetParquet.
    # how?  - Returns (train, test) in that order.
    return (PetParquet(files["train"], train_tf),
            PetParquet(files["test"], test_tf))
    #endregion


@torch.no_grad()
def measure_accuracy(model, loader, device):
    """
    * what? - Returns the percentage of photos in a loader whose breed the model gets right.
    * why?  - Accuracy after every epoch shows whether training works and feeds the report.
    * how?  - Eval mode, bf16 autocast, argmax over the 37 scores, counts hits over total.
    """
    #region 1 - Count the hits
    # what? - Loops over the loader and accumulates hits and total.
    # how?  - Same bf16 autocast as training, so the measurement matches the training numerics.
    model.eval()
    hits = total = 0
    for photos, labels in loader:
        photos, labels = photos.to(device), labels.to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            guess = model(photos).argmax(dim=1)
        hits += (guess == labels).sum().item()
        total += labels.numel()
    #endregion

    #region 2 - Return the percentage
    # what? - Converts counts to a percentage.
    # how?  - 100 * hits / total.
    return 100 * hits / total
    #endregion


def main():
    """
    * what? - Runs the whole stage 1/2 training and writes the model, the head and relatorio.json.
    * why?  - Produces the AFTER weights whose difference from the BEFORE weights is the experiment data.
    * how?  - Parse flags, build data and model, train with a frozen-body warm-up, save and report.
    """
    #region 1 - Parse flags and seed
    # what? - Reads the command line and fixes the random seed.
    # how?  - torch.manual_seed before anything random, so a seed identifies a run.
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--output", default=MODEL_AFTER)
    ap.add_argument("--weight-decay", type=float, default=0.05, dest="weight_decay")
    ap.add_argument("--body-lr", type=float, default=BODY_LR, dest="body_lr")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--warmup-epochs", type=int, default=WARMUP_EPOCHS, dest="warmup_epochs")
    args = ap.parse_args()
    started_at = datetime.datetime.now().isoformat(timespec="seconds")

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    #endregion

    #region 2 - Prepare the data
    # what? - Builds datasets and loaders.
    # how?  - Train shuffles and drops the last partial batch; test keeps order.
    print("1) downloading and preparing the data...")
    train, test = build_data()
    print(f"   train: {len(train)} photos | test: {len(test)} photos")
    print(f"   classes: {NUM_CLASSES}")

    train_loader = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test, batch_size=64, shuffle=False,
                             num_workers=4, pin_memory=True)
    #endregion

    #region 3 - Build the model and the optimizer
    # what? - Creates the classifier, the optimizer groups, the schedule and the loss.
    # how?  - ALL weights train, body included, because the body is what we want to measure.
    print("\n2) building the model...")
    model = PetClassifier(MODEL_BEFORE, NUM_CLASSES).to(device)

    groups = [
        {"params": model.body.parameters(), "lr": args.body_lr},
        {"params": model.head.parameters(), "lr": HEAD_LR},
    ]
    optimizer = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
    total_steps = args.epochs * len(train_loader)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_steps)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

    initial_acc = measure_accuracy(model, test_loader, device)
    print(f"   accuracy before training: {initial_acc:.1f}%  (random head)")
    #endregion

    #region 4 - Train
    # what? - The epoch loop with the frozen-body warm-up.
    # how?  - The body only unfreezes after the head already knows how to read it.
    print("\n3) training...")
    print(f"   (the body stays frozen for the first {args.warmup_epochs} epoch(s),")
    print("    so the head learns to read before it can push the body)")
    start = time.time()
    accuracy_per_epoch = []
    for epoch in range(1, args.epochs + 1):
        body_free = epoch > args.warmup_epochs
        for weight in model.body.parameters():
            weight.requires_grad_(body_free)

        model.train()
        loss_sum = 0.0
        for photos, labels in train_loader:
            photos, labels = photos.to(device), labels.to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = loss_fn(model(photos), labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            schedule.step()
            loss_sum += loss.item()

        acc = measure_accuracy(model, test_loader, device)
        mean_loss = loss_sum / len(train_loader)
        state = "body free  " if body_free else "body frozen"
        accuracy_per_epoch.append(round(acc, 2))
        print(f"   epoch {epoch}/{args.epochs}  {state}  loss {mean_loss:.3f}  accuracy {acc:.1f}%")

    minutes = (time.time() - start) / 60
    final_acc = measure_accuracy(model, test_loader, device)
    #endregion

    #region 5 - Save the AFTER model
    # what? - Writes the body (same format as the original), the head and the report.
    # how?  - The report keys stay in Portuguese: the frozen reports of all stages use this schema.
    print(f"\n4) saving the 'after' model in {args.output}/ ...")
    target = Path(args.output)
    target.mkdir(parents=True, exist_ok=True)
    model.body.save_pretrained(target)
    torch.save(model.head.state_dict(), target / "cabeca.pt")

    report = {
        "modelo_antes": MODEL_BEFORE,
        "modelo_depois": args.output,
        "dataset": "oxford-iiit-pet",
        "num_classes": NUM_CLASSES,
        "epocas": args.epochs,
        "lote": BATCH_SIZE,
        "lr_corpo": args.body_lr,
        "lr_cabeca": HEAD_LR,
        "epocas_aquecer": args.warmup_epochs,
        "semente": args.seed,
        "acuracia_antes": round(initial_acc, 2),
        "acuracia_depois": round(final_acc, 2),
        "minutos_de_treino": round(minutes, 1),
        # fields added in stage 2 (stage 1 reports do not have them)
        "weight_decay": args.weight_decay,
        "label_smoothing": 0.1,
        "otimizador": "AdamW",
        "agenda_lr": "cosseno, um passo por lote",
        "precisao": "bf16 (autocast)",
        "acuracias_por_epoca": accuracy_per_epoch,
        "comando": " ".join(sys.argv),
        "iniciado_em": started_at,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "treinar_py_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (target / "relatorio.json").write_text(json.dumps(report, indent=2))
    #endregion

    #region 6 - Print the summary
    # what? - Tells where the BEFORE and AFTER models are.
    # how?  - The BEFORE folder is only mentioned, never written.
    print(f"\nDONE in {minutes:.1f} min")
    print(f"  accuracy: {initial_acc:.1f}%  ->  {final_acc:.1f}%")
    print(f"  the 'before' model stays untouched in {MODEL_BEFORE}/")
    print(f"  the 'after' model is in {args.output}/")
    #endregion


if __name__ == "__main__":
    main()
