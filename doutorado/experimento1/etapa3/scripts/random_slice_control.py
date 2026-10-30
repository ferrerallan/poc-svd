"""
* what? - STAGE 3 control for the transplant: puts back a RANDOM slice of the change with the same dimension as the sigma part, and measures accuracy.
* why?  - The sigma part lives in only 384 of the 147,456 dimensions of a 384 x 384 matrix; "sigma alone recovers 0%" could be true of ANY slice that small. This tells the two apart.
* how?  - Per seed and matrix: dW = W1 - W0; D = sigma part on the SVD basis of W0; D' = the same projection on a random orthogonal basis (U', V'). Transplants W0 + D and W0 + D' (natural energy and rescaled to ||D||) into the fine-tuned model and tests.

Variants measured per seed (all other parameters and the head are those of the fine-tuned model):
  nenhuma            W0 (sanity check against transplante_etapa3.json)
  so_sigma           W0 + D (sanity check against transplante_etapa3.json)
  fatia_aleatoria    W0 + D', D' = U' diag(diag(U'^T dW V')) V'^T with U', V' random orthonormal (QR of Gaussian), DRAWS draws per seed
  fatia_aleatoria_eq W0 + D' * ||D|| / ||D'||, the random slice rescaled to the energy of the sigma part
Run from the repository root:
    python3 -u doutorado/experimento1/etapa3/scripts/random_slice_control.py | tee doutorado/experimento1/etapa3/logs/controle_fatia_aleatoria.log
Output: doutorado/experimento1/etapa3/controle_fatia_aleatoria.json
"""
import json
import statistics as st
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import Dinov2Model

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config_etapa3 as C  # noqa: E402
import transplant as TP  # noqa: E402

DRAWS = 2
RANDOM_BASE_SEED = 5000
OUTPUT = C.STAGE3 / "controle_fatia_aleatoria.json"


def random_basis(m, n, generator):
    """
    * what? - Returns a random orthonormal pair U' (m x k) and V' (n x k), k = min(m, n).
    * why?  - A random basis with the same shape as the SVD basis of W0 defines a slice of the same dimension as the sigma part.
    * how?  - QR decomposition of Gaussian matrices; Q has orthonormal columns.
    """
    #region 1 - Draw and orthonormalize
    # what? - Two matrices with orthonormal columns.
    # how?  - torch.linalg.qr in reduced mode returns m x k and n x k factors.
    k = min(m, n)
    U = torch.linalg.qr(torch.randn(m, k, generator=generator))[0]
    V = torch.linalg.qr(torch.randn(n, k, generator=generator))[0]
    return U, V
    #endregion


def diagonal_part(dW, U, V):
    """
    * what? - Projects dW onto the span of the k rank-1 matrices u_i v_i^T of a basis pair.
    * why?  - With the SVD basis of W0 this is the sigma part D; with a random basis it is the control slice D'.
    * how?  - d_i = u_i^T dW v_i, then U diag(d) V^T.
    """
    #region 1 - Project
    # what? - The projected matrix.
    # how?  - Same formula transplant.py uses for D, written with V (n x k) instead of V^T.
    d = ((U.T @ dW) * V.T).sum(1)
    return (U * d) @ V.T
    #endregion


def accuracy_with(body, state, head, loader, matrices):
    """
    * what? - Loads the given 72 matrices into the body and returns the test accuracy, rounded to 2 decimals.
    * why?  - Every variant of the control is measured through this one path, the same one transplant.py uses.
    * how?  - Overwrites the entries of the state dict, reloads it and calls transplant.head_accuracy.
    """
    #region 1 - Swap the matrices and test
    # what? - Accuracy in percent.
    # how?  - The dtype of each entry is kept (bf16 or float32 as saved).
    for n, W in matrices.items():
        state[n] = W.to(state[n].dtype)
    body.load_state_dict(state)
    return round(TP.head_accuracy(body, head, loader), 2)
    #endregion


def main():
    """
    * what? - Runs the control for the 10 skin seeds and writes the JSON.
    * why?  - Decides whether the transplant result about the sigma part is information or an artifact of slice size.
    * how?  - Reuses the loader, head and accuracy function of transplant.py; one random generator per (seed, draw).
    """
    #region 1 - Load the original model and the matrix names
    # what? - W0 of the 72 matrices.
    # how?  - Same selection rule as transplant.py.
    original = load_file(str(C.ORIGINAL_MODEL / "model.safetensors"))
    names = [n for n, t in original.items() if t.dim() == 2 and "encoder.layer." in n]
    rows = []
    loader = n_classes = None
    #endregion

    #region 2 - Seed by seed: build the slices and test them
    # what? - Accuracy of nenhuma, so_sigma and the random slices (natural and energy-matched) per seed and draw.
    # how?  - The body is loaded once per seed; each variant replaces the 72 matrices in its state dict.
    for s in C.SEEDS:
        folder = C.stage_model(3, s).parent
        report = json.load(open(folder / "relatorio.json"))
        if loader is None:
            loader, n_classes = TP.test_loader_of("etapa3", report)
        fine_tuned = load_file(str(folder / "model.safetensors"))
        head = torch.nn.Linear(384, n_classes).to(TP.DEVICE)
        head.load_state_dict(torch.load(folder / "cabeca.pt", map_location=TP.DEVICE))
        head.eval()
        body = Dinov2Model.from_pretrained(str(folder)).to(TP.DEVICE)
        sd = body.state_dict()

        sigma_parts, base = {}, {}
        energy_sigma = []
        for n in names:
            W0, W1 = original[n].float(), fine_tuned[n].float()
            dW = W1 - W0
            U0, _, Vt0 = torch.linalg.svd(W0, full_matrices=False)
            D = diagonal_part(dW, U0, Vt0.T)
            sigma_parts[n] = (W0, dW, D)
            base[n] = W0
            energy_sigma.append((D.norm() ** 2 / dW.norm() ** 2).item())
        row = {"seed": s, "nenhuma": accuracy_with(body, sd, head, loader, base),
               "so_sigma": accuracy_with(body, sd, head, loader, {n: W0 + D for n, (W0, dW, D) in sigma_parts.items()}),
               "energy_fraction_sigma_pct": 100 * st.mean(energy_sigma), "draws": []}
        for k in range(DRAWS):
            gen = torch.Generator().manual_seed(RANDOM_BASE_SEED + 100 * s + k)
            natural, matched, energy_random = {}, {}, []
            for n, (W0, dW, D) in sigma_parts.items():
                U, V = random_basis(W0.shape[0], W0.shape[1], gen)
                Dr = diagonal_part(dW, U, V)
                energy_random.append((Dr.norm() ** 2 / dW.norm() ** 2).item())
                natural[n] = W0 + Dr
                matched[n] = W0 + Dr * (D.norm() / Dr.norm())
            row["draws"].append({"draw": k, "fatia_aleatoria": accuracy_with(body, sd, head, loader, natural),
                                 "fatia_aleatoria_eq": accuracy_with(body, sd, head, loader, matched),
                                 "energy_fraction_random_pct": 100 * st.mean(energy_random)})
        rows.append(row)
        print(json.dumps(row), flush=True)
        del body, fine_tuned, sigma_parts, base
        torch.cuda.empty_cache()
    #endregion

    #region 3 - Summarize and write
    # what? - Means over seeds (and draws) of every variant, plus the energy fractions.
    # how?  - Random-slice variants average over seeds x draws; the sanity variants over seeds.
    rand = [d["fatia_aleatoria"] for r in rows for d in r["draws"]]
    rand_eq = [d["fatia_aleatoria_eq"] for r in rows for d in r["draws"]]
    summary = {"nenhuma": st.mean(r["nenhuma"] for r in rows), "so_sigma": st.mean(r["so_sigma"] for r in rows),
               "fatia_aleatoria": {"mean": st.mean(rand), "std": st.stdev(rand), "min": min(rand), "max": max(rand)},
               "fatia_aleatoria_eq": {"mean": st.mean(rand_eq), "std": st.stdev(rand_eq), "min": min(rand_eq), "max": max(rand_eq)},
               "energy_fraction_sigma_pct": st.mean(r["energy_fraction_sigma_pct"] for r in rows),
               "energy_fraction_random_pct": st.mean(d["energy_fraction_random_pct"] for r in rows for d in r["draws"])}
    OUTPUT.write_text(json.dumps({"draws_per_seed": DRAWS, "random_base_seed": RANDOM_BASE_SEED, "rows": rows, "summary": summary}, indent=1))
    print("summary:", json.dumps(summary))
    print("written:", OUTPUT.relative_to(C.ROOT))
    #endregion


if __name__ == "__main__":
    main()
