"""
* what? - Aggregates the 10 seeds of stage 1 and the 45 seed-pair "noise" comparisons.
* why?  - Separates what fine-tuning changed from what any two trainings differ by, seed by seed.
* how?  - Reads the JSONs written by compare.py and computes the cosine between the changes (dW) from the safetensors.

Output: etapa1/resultado_10sementes.json (and a summary on the terminal).
Run from the repository root:  python3 doutorado/experimento1/etapa1/scripts/analyze_seeds.py
"""
import itertools
import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
import stages  # noqa: E402
from safetensors.torch import load_file  # noqa: E402

SEEDS = list(range(10))
MEASURES = ["tamanho_mudanca", "variacao_dos_valores", "giro_dos_vetores_graus", "concentracao"]


def noise_file(i, j):
    """
    * what? - Returns the comparison JSON between the trained models of seeds i and j.
    * why?  - Seed pair (0, 1) was stored as resultado_ruido.json before the 45-pair naming existed.
    * how?  - Special-cases (0, 1); every other pair is ruido_i_j.json inside the stage 1 comparison folder.
    """
    #region 1 - Resolve the file
    # what? - Applies the legacy naming rule.
    # how?  - Files are looked up in stages.COMPARISON_DIR[1].
    return stages.COMPARISON_DIR[1] / ("resultado_ruido.json" if (i, j) == (0, 1) else f"ruido_{i}_{j}.json")
    #endregion


def index_rows(rows):
    """
    * what? - Indexes the rows of a comparison JSON by (block, type).
    * why?  - The aggregation looks values up by matrix, not by list position.
    * how?  - Dictionary comprehension over the rows.
    """
    #region 1 - Build the index
    # what? - Maps (bloco, tipo) to the row.
    # how?  - The keys stay in Portuguese: they are the schema of the frozen JSONs.
    return {(x["bloco"], x["tipo"]): x for x in rows}
    #endregion


def tensor_name(block, kind):
    """
    * what? - Rebuilds the checkpoint tensor name from a block number and a matrix type.
    * why?  - Comparison rows use short types but the safetensors file uses full names.
    * how?  - Attention matrices and dense have their own prefixes; the MLP types are already the suffix.
    """
    #region 1 - Rebuild the name
    # what? - Chooses the right prefix.
    # how?  - query/key/value live under attention.attention, dense under attention.output.
    if kind in ("query", "key", "value"):
        return f"encoder.layer.{block}.attention.attention.{kind}.weight"
    if kind == "dense":
        return f"encoder.layer.{block}.attention.output.dense.weight"
    return f"encoder.layer.{block}.{kind}.weight"
    #endregion


def summary(values):
    """
    * what? - Returns mean, standard deviation, minimum and maximum of a list.
    * why?  - Every aggregated entry of the JSON has the same four statistics.
    * how?  - statistics module; the standard deviation is 0 for a single value.
    """
    #region 1 - Compute the statistics
    # what? - Builds the dictionary.
    # how?  - Keys media, dp, min, max, as in the frozen file.
    return {"media": st.mean(values), "dp": st.stdev(values) if len(values) > 1 else 0.0,
            "min": min(values), "max": max(values)}
    #endregion


def main():
    """
    * what? - Aggregates seeds and noise pairs and writes resultado_10sementes.json.
    * why?  - It is the stage 1 analysis table: adaptation versus noise, global, by type, by block and by matrix.
    * how?  - Loads 10 + 45 comparison files, computes the dW cosines, then summarizes at every level.
    """
    #region 1 - Load the comparison files
    # what? - Reads adaptation results (original -> seed) and noise results (seed i -> seed j).
    # how?  - cells is the sorted list of (block, type) of seed 0.
    adaptation = {s: index_rows(json.load(open(stages.comparison(1, s)))) for s in SEEDS}
    pairs = list(itertools.combinations(SEEDS, 2))
    noise = {p: index_rows(json.load(open(noise_file(*p)))) for p in pairs}
    cells = sorted(adaptation[0])
    #endregion

    #region 2 - Compute the cosine between the changes, straight from the weights
    # what? - Cosine of dW_i and dW_j for every matrix and pair of seeds.
    # how?  - dW is the flattened difference between the trained and the original matrix.
    original = load_file("models/dinov2-small/model.safetensors")
    models = {s: load_file(stages.model(1, s)) for s in SEEDS}
    cos = {}
    for (b, t) in cells:
        n = tensor_name(b, t)
        deltas = {s: (models[s][n].float() - original[n].float()).flatten() for s in SEEDS}
        for (i, j) in pairs:
            di, dj = deltas[i], deltas[j]
            cos[(b, t, i, j)] = (di @ dj / (di.norm() * dj.norm())).item()
    #endregion

    #region 3 - Summarize globally
    # what? - One adaptation value per seed and one noise value per pair, for each measure.
    # how?  - Mean over the 72 matrices first, then the four statistics over seeds or pairs.
    output = {"sementes": SEEDS, "pares": [list(p) for p in pairs], "global": {}, "por_tipo": {}, "por_bloco": {}, "celulas": []}

    for m in MEASURES:
        a = [st.mean(adaptation[s][c][m] for c in cells) for s in SEEDS]
        r = [st.mean(noise[p][c][m] for c in cells) for p in pairs]
        output["global"][m] = {"adaptacao": summary(a), "ruido": summary(r)}

    pair_cos = [st.mean(cos[(b, t, i, j)] for (b, t) in cells) for (i, j) in pairs]
    output["global"]["cosseno_entre_mudancas"] = summary(pair_cos)
    #endregion

    #region 4 - Summarize by matrix type and by block
    # what? - The same statistics restricted to one type or one block.
    # how?  - Cells are filtered by the type or the block index.
    for t in sorted({c[1] for c in cells}):
        cl = [c for c in cells if c[1] == t]
        output["por_tipo"][t] = {"concentracao": {
            "adaptacao": summary([st.mean(adaptation[s][c]["concentracao"] for c in cl) for s in SEEDS]),
            "ruido": summary([st.mean(noise[p][c]["concentracao"] for c in cl) for p in pairs])},
            "cosseno": summary([st.mean(cos[(c[0], c[1], i, j)] for c in cl) for (i, j) in pairs])}

    for b in range(12):
        cl = [c for c in cells if c[0] == b]
        output["por_bloco"][str(b)] = {"cosseno": summary([st.mean(cos[(c[0], c[1], i, j)] for c in cl) for (i, j) in pairs])}
    #endregion

    #region 5 - Summarize per matrix and write the file
    # what? - One record per (block, type) with every measure and the cosine.
    # how?  - The 72 records go into the celulas list of the output.
    for c in cells:
        record = {"bloco": c[0], "tipo": c[1]}
        for m in MEASURES:
            record[m] = {"adaptacao": summary([adaptation[s][c][m] for s in SEEDS]),
                         "ruido": summary([noise[p][c][m] for p in pairs])}
        record["cosseno"] = summary([cos[(c[0], c[1], i, j)] for (i, j) in pairs])
        output["celulas"].append(record)

    stages.TEN_SEEDS_JSON.write_text(json.dumps(output, indent=2))
    print("written:", stages.TEN_SEEDS_JSON)
    #endregion


if __name__ == "__main__":
    main()
