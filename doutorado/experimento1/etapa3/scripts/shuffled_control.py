"""
* what? - STAGE 3: control with SHUFFLED LABELS; compares the change of the 72 matrices across real and shuffled trainings.
* why?  - If real-vs-real agreement is higher than real-vs-shuffled, the change carries task signal and not only noise.
* how?  - Cosines for the whole change, the sigma part and the off-diagonal part over three groups of seed pairs.

Groups: real x real (45 pairs), shuffled x shuffled (45 pairs) and real x shuffled (100 pairs).
Output: etapa3/controle_embaralhado.json
Run from the repository root:  python3 doutorado/experimento1/etapa3/scripts/shuffled_control.py
"""
import itertools
import json
import statistics as st
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config_etapa3 as C  # noqa: E402


def cosine(x, y):
    """
    * what? - Cosine similarity between two tensors of any shape (flattened).
    * why?  - It is the agreement measure between two trainings.
    * how?  - Dot product of the flattened tensors over the product of the norms.
    """
    #region 1 - Compute the cosine
    # what? - Normalized dot product.
    # how?  - .item() returns a plain float.
    return (x.flatten() @ y.flatten() / (x.norm() * y.norm())).item()
    #endregion


def main():
    """
    * what? - Computes the agreement of the three seed-pair groups and writes controle_embaralhado.json.
    * why?  - The result shows that real trainings agree (0.25) while real and shuffled barely do (0.03).
    * how?  - Per matrix: split every dW into whole, sigma and off-diagonal parts, then average the pairwise cosines.
    """
    #region 1 - Load the models
    # what? - Original, real and shuffled trainings, and the sorted matrix names.
    # how?  - Names are sorted by block number, then by name.
    original = load_file(str(C.ORIGINAL_MODEL / "model.safetensors"))
    real = {s: load_file(str(C.stage_model(3, s))) for s in C.SEEDS}
    shuffled = {s: load_file(str(C.shuffled_model(s))) for s in C.SEEDS}
    names = sorted([n for n, t in original.items() if t.dim() == 2 and "encoder.layer." in n], key=lambda n: (int(n.split(".")[2]), n))
    accum = {g: {"inteira": [], "sigma": [], "fora": []} for g in ("real_real", "emb_emb", "real_emb")}
    displacement = {"real": [], "emb": []}
    #endregion

    #region 2 - Measure every matrix
    # what? - Displacement and pairwise agreement per matrix.
    # how?  - The sigma part is the diagonal of U0' dW V0; the off-diagonal part is dW minus the sigma part.
    for n in names:
        W0 = original[n].float()
        U0, S0, Vt0 = torch.linalg.svd(W0, full_matrices=False)

        def parts(W1):
            """
            * what? - Splits the change W1 - W0 into the whole change, the sigma vector and the off-diagonal part.
            * why?  - The three parts are compared separately between trainings.
            * how?  - d is diag(U0' dW V0), D is the matrix built from d, the rest is dW - D.
            """
            #region 1 - Split the change
            # what? - Returns the three parts in a dictionary.
            # how?  - Uses the SVD of W0 computed for the current matrix.
            dW = W1 - W0
            d = ((U0.T @ dW) * Vt0).sum(1)
            D = (U0 * d) @ Vt0
            return {"inteira": dW, "sigma": d, "fora": dW - D}
            #endregion

        R = {s: parts(real[s][n].float()) for s in C.SEEDS}
        Em = {s: parts(shuffled[s][n].float()) for s in C.SEEDS}
        displacement["real"].append(st.mean(100 * (R[s]["inteira"].norm() / W0.norm()).item() for s in C.SEEDS))
        displacement["emb"].append(st.mean(100 * (Em[s]["inteira"].norm() / W0.norm()).item() for s in C.SEEDS))
        for k in ("inteira", "sigma", "fora"):
            accum["real_real"][k].append(st.mean(cosine(R[i][k], R[j][k]) for i, j in itertools.combinations(C.SEEDS, 2)))
            accum["emb_emb"][k].append(st.mean(cosine(Em[i][k], Em[j][k]) for i, j in itertools.combinations(C.SEEDS, 2)))
            accum["real_emb"][k].append(st.mean(cosine(R[i][k], Em[j][k]) for i in C.SEEDS for j in C.SEEDS))
    #endregion

    #region 3 - Assemble and write the result
    # what? - Summary, per-matrix table, displacement and shuffled accuracies.
    # how?  - The keys stay in Portuguese: the page builder reads this schema.
    acc_shuffled = [json.load(open(C.shuffled_model(s).parent / "relatorio.json")) for s in C.SEEDS]
    output = {"cosseno_medio_72": {g: {k: st.mean(v) for k, v in d.items()} for g, d in accum.items()},
              "por_matriz": [{"bloco": int(n.split(".")[2]), "tipo": ".".join(n.split(".")[3:-1]).replace("attention.attention.", "").replace("attention.output.", ""),
                              **{f"{g}_{k}": accum[g][k][i] for g in accum for k in accum[g]}} for i, n in enumerate(names)],
              "deslocamento_medio_pct": {k: st.mean(v) for k, v in displacement.items()},
              "embaralhado_acuracias": [{"semente": r["semente"], "teste_antes": r["acuracia_antes"], "teste_depois": r["acuracia_depois"],
                                         "treino_por_epoca": r["acuracia_treino_por_epoca"]} for r in acc_shuffled]}
    (C.STAGE3 / "controle_embaralhado.json").write_text(json.dumps(output, indent=1))
    print(json.dumps(output["cosseno_medio_72"], indent=1))
    print("displacement:", output["deslocamento_medio_pct"])
    #endregion


if __name__ == "__main__":
    main()
