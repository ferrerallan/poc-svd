"""
* what? - Explores the Oxford-IIIT Pet dataset used in experiment 1: labels, counts per breed, photo sizes, examples.
* why?  - Before reading any fine-tuning result we must know exactly what the model saw and how balanced it is.
* how?  - Read-only over the parquet files; writes tables and figures only (never touches data or earlier stages).

Run from the repository root:
    python3 doutorado/experimento1/etapa1/scripts/oxfordpet_exploration.py
Outputs:
    doutorado/experimento1/etapa1/exploracao_oxfordpet/   rotulos.json, contagens.csv, resumo.json,
                                                          tabela_racas.html, exploracao.log, ambiente.json
    artigo/images/   15_pets_mosaico_rotulos.png, 16_pets_contagem_por_raca.png,
                     17_pets_variacao_dentro_da_raca.png, 18_pets_tamanhos_das_fotos.png
Requirements: pandas, pyarrow, pillow, matplotlib, numpy.
"""
import io
import json
import platform
import sys
import textwrap
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[4]
DATA = ROOT / "data/hf/data"
OUTPUT = Path(__file__).resolve().parents[1] / "exploracao_oxfordpet"
FIGURES = ROOT / "artigo/images"
DOG_COLOR, CAT_COLOR = "#c26a12", "#6b46c1"
TEXT_COLOR = "#1a1a1c"


class Tee:
    """
    * what? - Writes everything that is printed both to the screen and to the log file.
    * why?  - The experiment diary requires a log of everything that was run.
    * how?  - Forwards each write and flush to all destinations.
    """

    def __init__(self, *targets):
        """
        * what? - Stores the destinations.
        * why?  - Needed before the first write.
        * how?  - Keeps the file-like objects in a tuple.
        """
        #region 1 - Store the destinations
        # what? - Remembers the file objects.
        # how?  - A tuple; any object with write and flush works.
        self.targets = targets
        #endregion

    def write(self, text):
        """
        * what? - Writes text to every destination.
        * why?  - Makes the object usable as sys.stdout.
        * how?  - Loops over the stored targets.
        """
        #region 1 - Forward the text
        # what? - Duplicates the output.
        # how?  - One write call per target.
        for t in self.targets:
            t.write(text)
        #endregion

    def flush(self):
        """
        * what? - Flushes every destination.
        * why?  - sys.stdout must implement flush.
        * how?  - Loops over the stored targets.
        """
        #region 1 - Forward the flush
        # what? - Flushes all targets.
        # how?  - One flush call per target.
        for t in self.targets:
            t.flush()
        #endregion


def load():
    """
    * what? - Loads train and test tables and the list of label names.
    * why?  - The breed names live in the parquet header, not in the rows.
    * how?  - pandas reads both files; the huggingface metadata of the train file holds the ClassLabel names.
    """
    #region 1 - Read the two tables
    # what? - Loads train and test with a 'parte' column.
    # how?  - Same file naming as the HuggingFace mirror.
    parts = {}
    for part in ("train", "test"):
        file = DATA / f"{part}-00000-of-00001.parquet"
        df = pd.read_parquet(file)
        df["parte"] = part
        parts[part] = df
    #endregion

    #region 2 - Read the label names
    # what? - Breed names and cat/dog names.
    # how?  - The huggingface metadata is JSON inside the parquet schema.
    meta = json.loads(pq.ParquetFile(DATA / "train-00000-of-00001.parquet").schema_arrow.metadata[b"huggingface"])
    names = meta["info"]["features"]["label"]["names"]
    cat_dog_names = meta["info"]["features"]["label_cat_dog"]["names"]
    return parts, names, cat_dog_names
    #endregion


def measure_photos(df):
    """
    * what? - Measures width, height and file size in bytes of every photo.
    * why?  - The model needs 224 x 224; we must know how far the originals are from that.
    * how?  - Opens only the header of each image (PIL is lazy), without decoding pixels.
    """
    #region 1 - Measure every photo
    # what? - Collects the three lists.
    # how?  - Image.open on an in-memory buffer; .size reads the header only.
    widths, heights, sizes = [], [], []
    for cell in df["image"]:
        b = cell["bytes"]
        w, h = Image.open(io.BytesIO(b)).size
        widths.append(w), heights.append(h), sizes.append(len(b))
    return np.array(widths), np.array(heights), np.array(sizes)
    #endregion


def photo(df, row, side=None):
    """
    * what? - Returns a photo as an RGB image, optionally center-cropped to a square thumbnail.
    * why?  - Mosaics need thumbnails of the same size.
    * how?  - Decodes the bytes, crops the central square and resizes it to `side`.
    """
    #region 1 - Decode the photo
    # what? - Opens the JPEG bytes as RGB.
    # how?  - In-memory buffer; nothing is written to disk.
    im = Image.open(io.BytesIO(df.iloc[row]["image"]["bytes"])).convert("RGB")
    #endregion

    #region 2 - Crop to a square thumbnail when asked
    # what? - Center crop and resize.
    # how?  - The crop box is centered on the shorter side.
    if side:
        w, h = im.size
        s = min(w, h)
        im = im.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s)).resize((side, side), Image.LANCZOS)
    return im
    #endregion


def plot_mosaic(train, names, species):
    """
    * what? - Draws one photo per breed with its label number and name, colored by species.
    * why?  - Shows the reader what each of the 37 numbers means.
    * how?  - 6 x 7 grid, first training photo of each breed, wrapped titles, explanation text in a free cell.
    """
    #region 1 - Build the grid
    # what? - Draws the 37 thumbnails.
    # how?  - Violet titles are cats and orange titles are dogs.
    fig, axes = plt.subplots(6, 7, figsize=(14, 11.4), gridspec_kw={"hspace": 0.5, "wspace": 0.08})
    for k, ax in enumerate(axes.ravel()):
        ax.axis("off")
        if k >= 37:
            continue
        row = int(np.where(train["label"].values == k)[0][0])
        ax.imshow(photo(train, row, 180))
        color = CAT_COLOR if species[k] == 0 else DOG_COLOR
        ax.set_title(textwrap.fill(f"{k} · {names[k].replace('_', ' ')}", 17), fontsize=9, color=color, fontweight="bold", pad=3)
    #endregion

    #region 2 - Add the legend text and save
    # what? - Explanatory text in the first free cell, then the file.
    # how?  - Figure content stays in Portuguese because it is read by the article audience.
    axes.ravel()[38].text(0.0, 0.8, "Cada foto mostra uma raça.\nTítulo = número do rótulo · nome.\n\nroxo = gato\nlaranja = cão", fontsize=11.5,
                          va="top", color=TEXT_COLOR, transform=axes.ravel()[38].transAxes)
    fig.suptitle("As 37 classes do Oxford-IIIT Pet: o número ao lado do nome é o rótulo que o modelo vê", fontsize=13.5, y=0.95)
    fig.savefig(FIGURES / "15_pets_mosaico_rotulos.png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    #endregion


def plot_counts(count_train, count_test, names, species):
    """
    * what? - Draws the number of train and test photos per breed.
    * why?  - Shows that the classes are balanced, so accuracy cannot be gamed by guessing the commonest breed.
    * how?  - Paired bars per breed: strong color for train, light color for test.
    """
    #region 1 - Draw the bars
    # what? - Two bars per breed, colored by species.
    # how?  - The legend sits above the bars because they reach 100 and the axis goes to 135.
    fig, ax = plt.subplots(figsize=(14, 5.2))
    x = np.arange(37)
    colors = [CAT_COLOR if species[i] == 0 else DOG_COLOR for i in range(37)]
    ax.bar(x - 0.2, count_train.values, 0.4, color=colors, label="treino")
    ax.bar(x + 0.2, count_test.values, 0.4, color=colors, alpha=0.45, label="teste")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{i} {names[i].replace('_', ' ')}" for i in range(37)], rotation=75, ha="right", fontsize=8.5)
    ax.set_ylabel("número de fotos")
    ax.set_title("Fotos por raça: o conjunto é equilibrado (cerca de 100 de treino e 100 de teste por raça)", fontsize=12.5)
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color="#777"), plt.Rectangle((0, 0), 1, 1, color="#777", alpha=0.45),
                       plt.Rectangle((0, 0), 1, 1, color=CAT_COLOR), plt.Rectangle((0, 0), 1, 1, color=DOG_COLOR)],
              labels=["treino (cor forte)", "teste (cor clara)", "gato", "cão"], ncol=4, loc="upper center", fontsize=9.5, frameon=False)
    ax.set_ylim(0, 135)
    fig.tight_layout()
    fig.savefig(FIGURES / "16_pets_contagem_por_raca.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    #endregion


def plot_variation(train, names):
    """
    * what? - Shows eight photos of one breed (beagle) with their ids and sizes.
    * why?  - Makes clear that one label covers very different photos: poses, light, crops.
    * how?  - 2 x 4 grid of the first eight beagle photos of the training set.
    """
    #region 1 - Draw the eight photos
    # what? - Images with id, size and label number as the title.
    # how?  - Photos keep their original aspect ratio here (no square crop).
    beagle = names.index("beagle")
    idx = np.where(train["label"].values == beagle)[0][:8]
    fig, axes = plt.subplots(2, 4, figsize=(12.5, 6.6))
    for ax, i in zip(axes.ravel(), idx):
        im = photo(train, int(i))
        ax.imshow(im)
        ax.set_xticks([]), ax.set_yticks([])
        ax.set_title(f"{train.iloc[int(i)]['image_id']}\n{im.size[0]} × {im.size[1]} pixels · rótulo {beagle}", fontsize=9.5)
    fig.suptitle(f"Oito fotos de beagle: mesmo rótulo ({beagle}), fotos muito diferentes (poses, luz, tamanhos)", fontsize=12.5)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(FIGURES / "17_pets_variacao_dentro_da_raca.png", dpi=115, bbox_inches="tight")
    plt.close(fig)
    #endregion


def plot_sizes(widths, heights):
    """
    * what? - Draws width versus height of every photo and a histogram of the longer side.
    * why?  - Shows that originals vary in size and none is 224 x 224, so all are resized before the model.
    * how?  - Scatter with the 224 x 224 point marked, plus a histogram clipped at 800 pixels; counts the outliers.
    """
    #region 1 - Draw the two panels
    # what? - Scatter and histogram.
    # how?  - Photos with the longer side above 800 pixels are left out of both and counted in the note.
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    axes[0].scatter(widths, heights, s=5, alpha=0.25, color="#6b46c1")
    axes[0].plot([0, 800], [0, 800], color="#888", lw=1, ls="--")
    axes[0].scatter([224], [224], color="#c0392b", s=80, zorder=5, label="224 × 224 (o que o modelo recebe)")
    axes[0].set_xlim(0, 800), axes[0].set_ylim(0, 800)
    axes[0].set_xlabel("largura (pixels)"), axes[0].set_ylabel("altura (pixels)")
    axes[0].set_title("Cada ponto é uma foto original", fontsize=12)
    axes[0].legend(loc="upper left", fontsize=9.5, frameon=False)
    side = np.maximum(widths, heights)
    outside = int((side > 800).sum())
    print(f"Photos with the longer side above 800 pixels (left out of the size plots): {outside}")
    axes[0].text(790, 15, f"{outside} fotos maiores que 800 ficam fora do gráfico", ha="right", fontsize=9, color="#555")
    axes[1].hist(side[side <= 800], bins=40, color="#1d7a54")
    axes[1].set_xlabel("lado maior da foto (pixels), até 800"), axes[1].set_ylabel("número de fotos")
    axes[1].set_title("As fotos originais variam; todas serão redimensionadas", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIGURES / "18_pets_tamanhos_das_fotos.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    #endregion


def main():
    """
    * what? - Runs the whole exploration, prints the findings and writes tables and figures.
    * why?  - One command reproduces everything the article shows about the dataset.
    * how?  - Load, count, measure, draw and write, in that order, teeing the console to a log file.
    """
    #region 1 - Prepare the folders and the log
    # what? - Creates the output folders and redirects stdout to console plus log.
    # how?  - Tee duplicates every print into exploracao.log.
    OUTPUT.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    log = open(OUTPUT / "exploracao.log", "w", encoding="utf-8")
    sys.stdout = Tee(sys.__stdout__, log)
    #endregion

    #region 2 - Load the data and describe the files
    # what? - Reads both tables and prints the basic facts.
    # how?  - 'todo' is train plus test, used for size statistics.
    parts, names, cat_dog_names = load()
    train, test = parts["train"], parts["test"]
    everything = pd.concat([train, test], ignore_index=True)
    print(f"Files in {DATA.relative_to(ROOT)}")
    print(f"Columns: {list(train.columns[:4])}  (image = photo bytes; label = breed number; image_id = original file name;"
          f" label_cat_dog = 0 cat, 1 dog)")
    print(f"Train photos: {len(train)}   Test photos: {len(test)}   Total: {len(everything)}")
    print(f"Breeds (classes): {len(names)}\n")
    #endregion

    #region 3 - Count photos per breed and check the species convention
    # what? - Counts per breed, species per breed and the capital-letter rule of the original file names.
    # how?  - The assertion fails if a breed mixes cats and dogs.
    count_train = train["label"].value_counts().reindex(range(37)).fillna(0).astype(int)
    count_test = test["label"].value_counts().reindex(range(37)).fillna(0).astype(int)
    species = train.groupby("label")["label_cat_dog"].agg(lambda s: int(s.iloc[0]))
    assert (train.groupby("label")["label_cat_dog"].nunique() == 1).all(), "a breed mixes dogs and cats"
    capital_initial = train.groupby("label")["image_id"].agg(lambda s: bool(s.iloc[0][0].isupper()))
    matches = all((species[i] == 0) == capital_initial[i] for i in range(37))
    print("Naming convention of the original dataset: file name with a CAPITAL initial = cat, lowercase = dog ->",
          "holds for all 37 breeds" if matches else "does NOT hold")
    rows = []
    print(f"\n{'n':>2}  {'breed (label)':<28}{'species':<8}{'train':>7}{'test':>7}  example image_id")
    for i, name in enumerate(names):
        kind = cat_dog_names[species[i]]
        example = train[train["label"] == i]["image_id"].iloc[0]
        print(f"{i:>2}  {name:<28}{('cat' if kind == 'cat' else 'dog'):<8}{count_train[i]:>7}{count_test[i]:>7}  {example}")
        rows.append({"numero": i, "raca": name, "especie": "gato" if kind == "cat" else "cao",
                     "treino": int(count_train[i]), "teste": int(count_test[i]), "exemplo_image_id": example})
    pd.DataFrame(rows).to_csv(OUTPUT / "contagens.csv", index=False)
    (OUTPUT / "rotulos.json").write_text(json.dumps({str(i): n for i, n in enumerate(names)}, indent=1, ensure_ascii=False))
    #endregion

    #region 4 - Print the dataset statistics
    # what? - Species, balance and train/test overlap.
    # how?  - The overlap must be zero: a photo in both sets would inflate the accuracy.
    n_cat_breeds = int((species == 0).sum())
    print(f"\nSpecies: {n_cat_breeds} cat breeds and {37 - n_cat_breeds} dog breeds")
    print(f"Cat photos in train: {int((train['label_cat_dog'] == 0).sum())}   dog photos: {int((train['label_cat_dog'] == 1).sum())}")
    print(f"Photos per breed in train: minimum {count_train.min()}, maximum {count_train.max()}, mean {count_train.mean():.1f}")
    print(f"Photos per breed in test:  minimum {count_test.min()}, maximum {count_test.max()}, mean {count_test.mean():.1f}")
    ids_train, ids_test = set(train["image_id"]), set(test["image_id"])
    print(f"Photos present in both train and test: {len(ids_train & ids_test)} (must be 0)")
    chance = 1 / 37
    print(f"Accuracy of always guessing the same breed: {100 * count_test.max() / len(test):.1f}%;  random guess: {100 * chance:.1f}%")
    #endregion

    #region 5 - Measure the photo sizes
    # what? - Width, height and file size statistics.
    # how?  - Reads only image headers, so it is fast.
    widths, heights, sizes = measure_photos(everything)
    print(f"\nPhoto width:  min {widths.min()}, median {int(np.median(widths))}, max {widths.max()}")
    print(f"Photo height: min {heights.min()}, median {int(np.median(heights))}, max {heights.max()}")
    print(f"File size: median {int(np.median(sizes)) // 1024} KB")
    print(f"Photos exactly 224 x 224: {int(((widths == 224) & (heights == 224)).sum())} of {len(everything)}")
    print(f"Photos wider than tall: {int((widths > heights).sum())};  taller than wide: {int((heights > widths).sum())}")
    #endregion

    #region 6 - Draw the figures
    # what? - The four article figures.
    # how?  - White background so they read well in light and dark themes of the article.
    plt.rcParams.update({"font.family": "DejaVu Sans", "savefig.facecolor": "white", "figure.facecolor": "white", "axes.facecolor": "white"})
    plot_mosaic(train, names, species)
    plot_counts(count_train, count_test, names, species)
    plot_variation(train, names)
    plot_sizes(widths, heights)
    #endregion

    #region 7 - Write the HTML table and the summary files
    # what? - tabela_racas.html (fragment for the article), resumo.json and ambiente.json.
    # how?  - The table text stays in Portuguese because it is pasted into the article.
    rows_html = "\n".join(
        f"<tr><td class=\"n\">{r['numero']}</td><td>{r['raca'].replace('_', ' ')}</td><td>{r['especie']}</td>"
        f"<td class=\"n\">{r['treino']}</td><td class=\"n\">{r['teste']}</td></tr>" for r in rows)
    (OUTPUT / "tabela_racas.html").write_text(
        "<div class=\"tbl\"><table>\n<tr><th>Rótulo</th><th>Raça</th><th>Espécie</th><th>Fotos de treino</th><th>Fotos de teste</th></tr>\n"
        + rows_html + "\n</table></div>\n", encoding="utf-8")
    summary = {"treino": len(train), "teste": len(test), "racas": 37, "racas_gato": n_cat_breeds, "racas_cao": 37 - n_cat_breeds,
               "treino_min_por_raca": int(count_train.min()), "treino_max_por_raca": int(count_train.max()),
               "teste_min_por_raca": int(count_test.min()), "teste_max_por_raca": int(count_test.max()),
               "largura_min_med_max": [int(widths.min()), int(np.median(widths)), int(widths.max())],
               "altura_min_med_max": [int(heights.min()), int(np.median(heights)), int(heights.max())],
               "fotos_em_treino_e_teste": len(ids_train & ids_test)}
    (OUTPUT / "resumo.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    (OUTPUT / "ambiente.json").write_text(json.dumps({"python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__,
                                                      "matplotlib": matplotlib.__version__, "plataforma": platform.platform()}, indent=1))
    print(f"\nWritten to {OUTPUT.relative_to(ROOT)} and {FIGURES.relative_to(ROOT)}")
    log.close()
    sys.stdout = sys.__stdout__
    #endregion


if __name__ == "__main__":
    main()
