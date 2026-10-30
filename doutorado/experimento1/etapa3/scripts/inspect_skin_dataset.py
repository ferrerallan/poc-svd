"""
* what? - STAGE 3, session 12 of the diary: inspects the skin lesion dataset (HuggingFace Pranavkpba2000/skin_cancer_small_dataset).
* why?  - The dataset card gives no origin and no class names; this script records what the files themselves contain and checks for repeated photos.
* how?  - Reads class names from the parquet metadata, counts photos per class (full files and the sub-sample used in training), checks image size, looks for exact and near duplicates, and draws one photo per class.

Duplicates: exact = same bytes (sha256); near = same 9 x 8 difference hash (dHash) of the photo shrunk to grayscale.
Near duplicates catch re-saved copies; they do NOT catch rotated, flipped or cropped copies.
Run from the repository root:
    python3 doutorado/experimento1/etapa3/scripts/inspect_skin_dataset.py | tee doutorado/experimento1/etapa3/logs/inspecao_pele.log
Outputs: doutorado/experimento1/etapa3/inspecao_pele.json and artigo/images/19_pele_exemplos.png
"""
import collections
import hashlib
import io
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config_etapa3 as C  # noqa: E402

TRAIN = C.DATA / "pele/data/train-00000-of-00001-2b84ce2361bc8c42.parquet"
TEST = C.DATA / "pele/data/test-00000-of-00001-3a9b253c406a4af7.parquet"
OUTPUT = C.STAGE3 / "inspecao_pele.json"
FIGURE = C.ROOT / "artigo/images/19_pele_exemplos.png"


def image_bytes(cell):
    """
    * what? - Returns the raw JPEG/PNG bytes of an image cell.
    * why?  - The parquet stores each photo as a dict with the file bytes.
    * how?  - Takes the "bytes" key when the cell is a dict, otherwise the cell itself.
    """
    #region 1 - Extract the bytes
    # what? - A bytes object.
    # how?  - Matches how train_etapa3.py reads the same column.
    return cell["bytes"] if isinstance(cell, dict) else cell
    #endregion


def difference_hash(raw):
    """
    * what? - Returns a 64-bit fingerprint of how the brightness changes from left to right in a tiny version of the photo.
    * why?  - Two re-saved copies of the same photo get the same fingerprint even when their bytes differ.
    * how?  - Grayscale, resize to 9 x 8, compare each pixel with its right neighbour.
    """
    #region 1 - Compute the fingerprint
    # what? - A 64-character string of 0 and 1.
    # how?  - Pixel > right neighbour gives 1.
    small = np.asarray(Image.open(io.BytesIO(raw)).convert("L").resize((9, 8), Image.LANCZOS), dtype=np.int16)
    return "".join("1" if v else "0" for v in (small[:, :-1] > small[:, 1:]).ravel())
    #endregion


def main():
    """
    * what? - Prints the facts about the dataset, writes the JSON and draws the example figure.
    * why?  - Gives session 12 a measured description of its dataset.
    * how?  - Seven steps: names, counts, sub-sample counts, sizes, exact duplicates, near duplicates, figure.
    """
    #region 1 - Class names and counts
    # what? - The 8 names from the parquet metadata and the photos per class in each file.
    # how?  - The names are in the HuggingFace feature description stored inside the parquet, not in the dataset card.
    meta = json.loads(pq.ParquetFile(TRAIN).schema_arrow.metadata[b"huggingface"])
    names = meta["info"]["features"]["label"]["names"]
    tr, te = pd.read_parquet(TRAIN), pd.read_parquet(TEST)
    count_tr, count_te = collections.Counter(tr.label), collections.Counter(te.label)
    print("classes (from the parquet metadata):", names)
    print(f"{'label':>5} {'name':>5} {'train':>6} {'test':>5}")
    for k, n in enumerate(names):
        print(f"{k:>5} {n:>5} {count_tr[k]:>6} {count_te[k]:>5}")
    print(f"total train {len(tr)}, test {len(te)}")
    #endregion

    #region 2 - The sub-sample used in training
    # what? - Photos per class among the 3,680 chosen for training.
    # how?  - Same generator as train_etapa3.build_data: default_rng(SUBSAMPLE_SEED), first draw is the train sample.
    g = np.random.default_rng(C.SUBSAMPLE_SEED)
    chosen = np.sort(g.choice(len(tr), C.MAX_TRAIN, replace=False))
    count_sub = collections.Counter(tr.label.iloc[chosen])
    print("sub-sample used in training (class: photos):", {names[k]: count_sub[k] for k in range(8)})
    print(f"largest class in the test file: {max(count_te.values()) / len(te):.2%}")
    #endregion

    #region 3 - Image sizes
    # what? - Set of (width, height, mode) over every photo.
    # how?  - Opens each photo; 14,200 small files are quick to read.
    raw_tr = [image_bytes(c) for c in tr.image]
    raw_te = [image_bytes(c) for c in te.image]
    sizes = collections.Counter((Image.open(io.BytesIO(b)).size, Image.open(io.BytesIO(b)).mode) for b in raw_tr + raw_te)
    print("image sizes:", {f"{s[0][0]}x{s[0][1]} {s[1]}": n for s, n in sizes.items()})
    #endregion

    #region 4 - Exact duplicates
    # what? - Photos with identical bytes inside the train file and between train and test.
    # how?  - sha256 of the bytes.
    h_tr = [hashlib.sha256(b).hexdigest() for b in raw_tr]
    h_te = [hashlib.sha256(b).hexdigest() for b in raw_te]
    exact_in_train = len(h_tr) - len(set(h_tr))
    exact_cross = len(set(h_tr) & set(h_te))
    print(f"exact duplicates: inside train {exact_in_train}; test photos also in train {sum(h in set(h_tr) for h in h_te)}")
    #endregion

    #region 5 - Near duplicates
    # what? - Test photos whose fingerprint also appears among the train photos, and how often that fingerprint has the same label.
    # how?  - dHash of every photo; a dictionary from fingerprint to the train labels that have it.
    d_tr = [difference_hash(b) for b in raw_tr]
    d_te = [difference_hash(b) for b in raw_te]
    by_hash = collections.defaultdict(list)
    for d, y in zip(d_tr, tr.label):
        by_hash[d].append(y)
    hit = [(i, by_hash[d]) for i, d in enumerate(d_te) if d in by_hash]
    same_label = sum(1 for i, labels in hit if te.label.iloc[i] in labels)
    near_in_train = len(d_tr) - len(set(d_tr))
    print(f"near duplicates: inside train {near_in_train}; test photos with a train photo of the same fingerprint {len(hit)} "
          f"(same label in {same_label}); this check misses rotated, flipped or cropped copies")
    #endregion

    #region 6 - One photo per class
    # what? - A figure with one photo of each of the 8 classes.
    # how?  - The first photo of each class in the train file; 2 rows of 4.
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 4, figsize=(10, 5.4))
    for k, ax in enumerate(axes.ravel()):
        first = int(np.flatnonzero(tr.label.to_numpy() == k)[0])
        ax.imshow(Image.open(io.BytesIO(raw_tr[first])).convert("RGB"))
        ax.set_title(f"{k} = {names[k]}", fontsize=12)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(FIGURE, dpi=130)
    #endregion

    #region 7 - Write the JSON
    # what? - Every number printed above.
    # how?  - Keys in English because this file is new.
    OUTPUT.write_text(json.dumps({
        "repo": "Pranavkpba2000/skin_cancer_small_dataset", "class_names": names,
        "train_per_class": [count_tr[k] for k in range(8)], "test_per_class": [count_te[k] for k in range(8)],
        "train_subsample_per_class": [count_sub[k] for k in range(8)], "train_rows": len(tr), "test_rows": len(te),
        "largest_test_class_pct": 100 * max(count_te.values()) / len(te),
        "image_sizes": {f"{s[0][0]}x{s[0][1]} {s[1]}": n for s, n in sizes.items()},
        "exact_duplicates_inside_train": exact_in_train, "exact_duplicates_test_in_train": sum(h in set(h_tr) for h in h_te),
        "near_duplicates_inside_train": near_in_train, "near_duplicates_test_with_train": len(hit), "near_duplicates_same_label": same_label,
    }, indent=2))
    print("written:", OUTPUT.relative_to(C.ROOT), "and", FIGURE.relative_to(C.ROOT))
    #endregion


if __name__ == "__main__":
    main()
