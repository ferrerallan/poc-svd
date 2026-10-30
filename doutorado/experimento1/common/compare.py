"""
* what? - Compares the 72 weight matrices of DINOv2 before and after fine-tuning, one matrix at a time.
* why?  - Answers the thesis question per matrix: how much moved, did the singular values change, did the directions?
* how?  - SVD of the BEFORE matrix as reference; measures displacement, sigma variation and direction turn; writes a JSON.

For each matrix it answers three questions:
  1. did it move a lot? (size of the displacement)
  2. did the singular VALUES change? (how much each direction is amplified)
  3. did the singular VECTORS change? (where each direction points)

Run from the repository root:
    python3 doutorado/experimento1/common/compare.py --before <model.safetensors> --after <model.safetensors> --output <file.json>
"""
import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file

# ------------------------------------------------------------------
BEFORE = "models/dinov2-small/model.safetensors"
AFTER = "models/dinov2-small-pets/model.safetensors"
OUTPUT = "doutorado/experimento1/etapa1/comparacoes/resultado.json"

TOP = 10   # how many principal directions to look at closely
# ------------------------------------------------------------------


def matrices_2d(weights):
    """
    * what? - Keeps only the 2-D matrices of the 12 transformer blocks, as float32.
    * why?  - Bias vectors, norms and embeddings are not the matrices the thesis analyzes.
    * how?  - Filters the state dict by tensor rank 2 and by the 'encoder.layer.' name prefix.
    """
    #region 1 - Filter the matrices
    # what? - Builds the name -> float matrix dictionary.
    # how?  - float() so the SVD runs in float32 even if the file stores another dtype.
    return {n: t.float() for n, t in weights.items()
            if t.dim() == 2 and "encoder.layer." in n}
    #endregion


def nickname(name):
    """
    * what? - Turns a tensor name like encoder.layer.3.mlp.fc1.weight into (3, 'mlp.fc1').
    * why?  - Results are indexed by block number and matrix type, not by the long checkpoint name.
    * how?  - Splits on dots, takes the block index and strips the attention prefixes of the type.
    """
    #region 1 - Split the name
    # what? - Extracts block number and matrix type.
    # how?  - Index 2 is the block; the type is everything between the block and the final 'weight'.
    parts = name.split(".")
    block = int(parts[2])
    kind = ".".join(parts[3:-1]).replace("attention.attention.", "").replace("attention.output.", "")
    return block, kind
    #endregion


def compare_one(W0, W1):
    """
    * what? - Computes every measure for one matrix pair (before W0, after W1).
    * why?  - These measures are the raw data of the whole thesis: displacement, sigma change, direction turn.
    * how?  - SVD of W0 as reference, sigma variation, abs-cosine turn of U, and the diagonal share of U0' dW V0.
    """
    #region 1 - Measure the displacement
    # what? - Size of the change relative to the original matrix.
    # how?  - Frobenius norm of the difference over the norm of W0.
    dW = W1 - W0
    change_size = (dW.norm() / W0.norm()).item()
    #endregion

    #region 2 - Decompose before and after
    # what? - SVD of W0 (the reference basis) and the singular values of W1.
    # how?  - full_matrices=False keeps the economy-size factors.
    U0, S0, Vt0 = torch.linalg.svd(W0, full_matrices=False)
    S1 = torch.linalg.svdvals(W1)
    #endregion

    #region 3 - Measure the sigma variation and the direction turn
    # what? - Mean percentage change of the top sigmas and mean turn of the top left singular vectors.
    # how?  - abs() on the cosine is the fix for the sign trap: u and -u are the same direction.
    sigma_variation = ((S1[:TOP] - S0[:TOP]).abs() / S0[:TOP]).mean().item()

    U1, _, Vt1 = torch.linalg.svd(W1, full_matrices=False)
    cos = (U0[:, :TOP] * U1[:, :TOP]).sum(dim=0).abs().clamp(max=1.0)
    turn_degrees = torch.rad2deg(torch.arccos(cos)).mean().item()
    #endregion

    #region 4 - Split the displacement into sigma part and rotation part
    # what? - Share of the displacement energy that lies on the diagonal of the BEFORE basis.
    # how?  - Projects dW on the BEFORE basis; the diagonal is pure sigma change, the rest is rotation.
    M = U0.T @ dW @ Vt0.T
    total_energy = (M ** 2).sum()
    diagonal_energy = (M.diagonal() ** 2).sum()
    values_share = (diagonal_energy / total_energy).item()

    # a random displacement would give share = 1 / larger_side; dividing by it, 1 means "same as noise"
    larger_side = max(W0.shape)
    concentration = values_share * larger_side
    #endregion

    #region 5 - Count stuck singular values
    # what? - Number of top sigmas almost equal to the next one.
    # how?  - Trap 2: nearly equal singular values make their directions unstable.
    gap = ((S0[:TOP] - S0[1:TOP + 1]) / S0[:TOP])
    stuck = int((gap < 0.02).sum())
    #endregion

    #region 6 - Assemble the result
    # what? - Rounds and names the measures.
    # how?  - The keys stay in Portuguese: they are the schema of every frozen JSON.
    return {
        "tamanho_mudanca": round(change_size, 5),
        "variacao_dos_valores": round(sigma_variation, 5),
        "giro_dos_vetores_graus": round(turn_degrees, 3),
        "fatia_dos_valores": round(values_share, 6),
        "concentracao": round(concentration, 2),
        "sigmas_colados_no_topo": stuck,
    }
    #endregion


def main():
    """
    * what? - Compares two safetensors files matrix by matrix, prints a table and writes the JSON.
    * why?  - One call turns a before/after pair of models into the measures the analysis scripts consume.
    * how?  - Loads both files, pairs the matrices by name, runs compare_one on each and dumps the list.
    """
    #region 1 - Parse flags and check the files
    # what? - Reads the paths and aborts early if a file is missing.
    # how?  - SystemExit gives a clean message instead of a traceback.
    ap = argparse.ArgumentParser()
    ap.add_argument("--before", default=BEFORE)
    ap.add_argument("--after", default=AFTER)
    ap.add_argument("--output", default=OUTPUT)
    args = ap.parse_args()

    for path in (args.before, args.after):
        if not Path(path).exists():
            raise SystemExit(f"could not find {path}")
    #endregion

    #region 2 - Load and pair the matrices
    # what? - Reads both models and keeps the matrices present in both.
    # how?  - Sorted by (block, type) so the output order is stable.
    print(f"BEFORE: {args.before}")
    print(f"AFTER:  {args.after}")
    before = matrices_2d(load_file(args.before))
    after = matrices_2d(load_file(args.after))
    common = sorted(set(before) & set(after), key=nickname)
    print(f"comparing {len(common)} matrices\n")
    #endregion

    #region 3 - Measure every matrix
    # what? - Runs compare_one on each pair.
    # how?  - Each entry carries block, type and the original tensor name.
    results = []
    for name in common:
        block, kind = nickname(name)
        measures = compare_one(before[name], after[name])
        results.append({"bloco": block, "tipo": kind, "nome": name, **measures})
    #endregion

    #region 4 - Print the table
    # what? - Human-readable summary on the terminal.
    # how?  - Percentages for sizes, degrees for the turn.
    header = f"{'block':>5} {'type':<12} {'moved':>8} {'values':>9} {'turn':>8} {'concentr':>9}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(f"{r['bloco']:>5} {r['tipo']:<12} "
              f"{100*r['tamanho_mudanca']:>7.2f}% "
              f"{100*r['variacao_dos_valores']:>8.2f}% "
              f"{r['giro_dos_vetores_graus']:>7.2f}o "
              f"{r['concentracao']:>9.1f}")
    #endregion

    #region 5 - Write the JSON
    # what? - Saves the list of per-matrix results.
    # how?  - Plain json.dumps with indent 2, same as the frozen files.
    Path(args.output).write_text(json.dumps(results, indent=2))
    print(f"\nwritten to {args.output}")
    #endregion


if __name__ == "__main__":
    main()
