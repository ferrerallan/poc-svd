"""
* what? - STAGE 3: the CORRECTED RULER, applied to the models of stages 1, 2, 3 and to the shuffled-labels control.
* why?  - The stage 1-2 measures were inflated or structural; this ruler adds subspace angles and an isotropic-noise zero.
* how?  - Per matrix (72) and seed (10): sigma, subspace, matrix and noise measures plus agreement between seeds.

Read-only over the models. Output: etapa3/regua_<set>.json
Per matrix and seed:
  weights    relative change of ALL singular values (full spectrum) and of the top 10;
             overall scale (signed mean) and differential part, over the top 10
  directions principal angles between the top-k SUBSPACES (k = 10, 50, 100) of U0 and U1: independent of which vector is
             which and of sign, and stable when neighbouring singular values are almost equal;
             for continuity, the vector-by-vector top-10 turn of stages 1 and 2
  matrix     relative displacement ||W1-W0||/||W0||; share of the change energy that is the "sigma part" (diagonal), computed
             with the FULL norm of dW (stages 1-2 used the projection on span(U0), which drops energy in rectangular matrices)
  zero       the same measures for isotropic Gaussian noise with the SAME Frobenius norm as the real change of that matrix/seed
  agreement  between seeds (cosine, 45 pairs): whole change; sigma part; off-diagonal part; vector of change of all sigmas
Run from the repository root:  python3 doutorado/experimento1/etapa3/scripts/measure_ruler.py --set etapa1|etapa2|etapa3|embaralhado
"""
import argparse
import itertools
import json
import statistics as st
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config_etapa3 as C  # noqa: E402

KS = (10, 50, 100)
TOP = 10


def model_path(model_set, seed):
    """
    * what? - Returns the model path of a model set (stage 1, 2, 3 or the shuffled control) and seed.
    * why?  - The same measurement runs over four sets of trainings.
    * how?  - The shuffled set has its own helper; the others take the stage number from the set name.
    """
    #region 1 - Resolve the path
    # what? - Picks the helper of the set.
    # how?  - The last character of etapa1/etapa2/etapa3 is the stage number.
    if model_set == "embaralhado":
        return C.shuffled_model(seed)
    return C.stage_model(int(model_set[-1]), seed)
    #endregion


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


def matrix_type(name):
    """
    * what? - Returns the matrix type of a tensor name, such as 'mlp.fc1' or 'query'.
    * why?  - Cells are labeled by block number and type.
    * how?  - Joins the name parts between the block number and 'weight' and strips the attention prefixes.
    """
    #region 1 - Extract the type
    # what? - Cuts the checkpoint name down to the type.
    # how?  - Same rule as nickname() in compare.py.
    return ".".join(name.split(".")[3:-1]).replace("attention.attention.", "").replace("attention.output.", "")
    #endregion


def principal_angles(U0k, U1k):
    """
    * what? - Returns the mean and the largest principal angle (degrees) between two subspaces.
    * why?  - Subspace angles do not depend on vector order or sign and stay stable for nearly equal sigmas.
    * how?  - arccos of the singular values of U0k' U1k, clamped to 1.
    """
    #region 1 - Compute the angles
    # what? - Mean and maximum angle.
    # how?  - svdvals of the k x k overlap matrix.
    cos = torch.linalg.svdvals(U0k.T @ U1k).clamp(max=1.0)
    angles = torch.rad2deg(torch.arccos(cos))
    return angles.mean().item(), angles.max().item()
    #endregion


def measures(W0, U0, S0, Vt0, W1):
    """
    * what? - Computes all measures of one matrix for one version W1 (real or noise); returns them and the change parts.
    * why?  - The same function measures the real change and the isotropic-noise zero, so they are comparable.
    * how?  - Sigma changes, vector turn, subspace angles for each k, sigma/off-diagonal split of dW with its FULL norm.
    """
    #region 1 - Decompose W1 and measure the sigma side
    # what? - Relative sigma changes and the displacement.
    # how?  - rel is (S1 - S0) / S0 over the full spectrum.
    U1, S1, _ = torch.linalg.svd(W1, full_matrices=False)
    dW = W1 - W0
    rel = (S1 - S0) / S0
    m = {"desloc_pct": 100 * (dW.norm() / W0.norm()).item(),
         "sigma_top10_abs_pct": 100 * rel[:TOP].abs().mean().item(),
         "sigma_todos_abs_pct": 100 * rel.abs().mean().item(),
         "sigma_11_em_diante_abs_pct": 100 * rel[TOP:].abs().mean().item(),
         "escala_geral_top10_pct": 100 * rel[:TOP].mean().item(),
         "diferencial_top10_pct": 100 * (rel[:TOP] - rel[:TOP].mean()).abs().mean().item(),
         "giro_vetor_top10_graus": torch.rad2deg(torch.arccos((U0[:, :TOP] * U1[:, :TOP]).sum(0).abs().clamp(max=1))).mean().item()}
    #endregion

    #region 2 - Measure the subspace angles
    # what? - Mean and largest principal angle for each k.
    # how?  - Keys are sub<k>_medio_graus and sub<k>_maior_graus (frozen schema).
    for k in KS:
        mean_angle, max_angle = principal_angles(U0[:, :k], U1[:, :k])
        m[f"sub{k}_medio_graus"] = mean_angle
        m[f"sub{k}_maior_graus"] = max_angle
    #endregion

    #region 3 - Split the change into sigma part and the rest
    # what? - Diagonal (sigma) energy share with the FULL norm and with the old projection.
    # how?  - d = diag(U0' dW V0); D is the sigma part; the projection M_proj is what stages 1-2 used.
    d = ((U0.T @ dW) * Vt0).sum(1)
    D = (U0 * d) @ Vt0
    M_proj = (U0.T @ dW @ Vt0.T)
    m["fracao_sigma_completa_pct"] = 100 * ((D.norm() ** 2) / (dW.norm() ** 2)).item()
    m["fracao_sigma_projetada_pct"] = 100 * ((d ** 2).sum() / (M_proj ** 2).sum()).item()
    m["energia_fora_do_span_pct"] = 100 * (1 - (M_proj.norm() ** 2 / dW.norm() ** 2)).item()
    m["concentracao"] = m["fracao_sigma_completa_pct"] / 100 * max(W0.shape)
    return m, {"dW": dW, "d": d, "O": dW - D, "rel": rel}
    #endregion


def main():
    """
    * what? - Measures one model set with the corrected ruler and writes regua_<set>.json.
    * why?  - The ruler is applied to stages 1, 2, 3 and to the shuffled control so all are comparable.
    * how?  - Per matrix: real measures per seed, isotropic-noise measures with the same norm, agreement over 45 pairs.
    """
    #region 1 - Parse flags and load the models
    # what? - Reads the model set and loads the original and the 10 trained models.
    # how?  - The noise generator is seeded with 2024 and drawn in a fixed matrix and seed order.
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True, choices=["etapa1", "etapa2", "etapa3", "embaralhado"], dest="model_set")
    a = ap.parse_args()
    original = load_file(str(C.ORIGINAL_MODEL / "model.safetensors"))
    models = {s: load_file(str(model_path(a.model_set, s))) for s in C.SEEDS}
    names = sorted([n for n, t in original.items() if t.dim() == 2 and "encoder.layer." in n], key=lambda n: (int(n.split(".")[2]), n))
    pairs = list(itertools.combinations(C.SEEDS, 2))
    gen = torch.Generator().manual_seed(2024)
    #endregion

    #region 2 - Measure every matrix
    # what? - Real and noise measures per seed, plus the agreement between seeds.
    # how?  - The noise E has the same Frobenius norm as the real change of that matrix and seed.
    cells, agreement = [], {"inteira": [], "sigma": [], "fora": [], "espectro": []}
    for n in names:
        W0 = original[n].float()
        U0, S0, Vt0 = torch.linalg.svd(W0, full_matrices=False)
        real, zero, parts = [], [], {}
        for s in C.SEEDS:
            W1 = models[s][n].float()
            m, p = measures(W0, U0, S0, Vt0, W1)
            real.append(m)
            parts[s] = p
            E = torch.randn(W0.shape, generator=gen)
            E *= p["dW"].norm() / E.norm()
            mz, _ = measures(W0, U0, S0, Vt0, W0 + E)
            zero.append(mz)
        cc = {k: st.mean(cosine(parts[i][key], parts[j][key]) for i, j in pairs)
              for k, key in (("inteira", "dW"), ("sigma", "d"), ("fora", "O"), ("espectro", "rel"))}
        for k in agreement:
            agreement[k].append(cc[k])
        cell = {"bloco": int(n.split(".")[2]), "tipo": matrix_type(n), "formato": list(W0.shape), "concordancia": cc}
        for key in real[0]:
            v = [r[key] for r in real]
            z = [r[key] for r in zero]
            cell[key] = {"media": st.mean(v), "dp": st.stdev(v), "zero_media": st.mean(z), "zero_dp": st.stdev(z)}
        cells.append(cell)
    #endregion

    #region 3 - Aggregate globally and attach the accuracies
    # what? - Mean of the 72 matrices for every measure, and the accuracy of each seed.
    # how?  - Accuracies come from the relatorio.json that sits next to each model.
    glob = {}
    for key in [k for k in cells[0] if isinstance(cells[0][k], dict) and "media" in cells[0][k]]:
        glob[key] = {"media": st.mean(c[key]["media"] for c in cells), "zero_media": st.mean(c[key]["zero_media"] for c in cells)}
    glob["concordancia"] = {k: st.mean(v) for k, v in agreement.items()}
    accuracies = []
    for s in C.SEEDS:
        r = json.load(open(model_path(a.model_set, s).parent / "relatorio.json"))
        accuracies.append({"semente": s, "antes": r["acuracia_antes"], "depois": r["acuracia_depois"],
                           "treino_final": (r.get("acuracia_treino_por_epoca") or [None])[-1]})
    #endregion

    #region 4 - Write the result and print the summary
    # what? - Saves regua_<set>.json and prints one summary line.
    # how?  - The keys stay in Portuguese: the page builder reads this schema.
    output = {"conjunto": a.model_set, "n_sementes": len(C.SEEDS), "pares": len(pairs), "ks": KS, "topo": TOP,
              "global": glob, "celulas": cells, "acuracias": accuracies}
    (C.STAGE3 / f"regua_{a.model_set}.json").write_text(json.dumps(output, indent=1))
    g = glob
    print(f"{a.model_set}: displacement {g['desloc_pct']['media']:.3f}% | sigma top10 {g['sigma_top10_abs_pct']['media']:.4f}% (zero {g['sigma_top10_abs_pct']['zero_media']:.4f}) | "
          f"sigma all {g['sigma_todos_abs_pct']['media']:.4f}% | sub10 {g['sub10_medio_graus']['media']:.3f} deg (zero {g['sub10_medio_graus']['zero_media']:.3f}) | "
          f"sub50 {g['sub50_medio_graus']['media']:.3f} | vector turn {g['giro_vetor_top10_graus']['media']:.3f} (zero {g['giro_vetor_top10_graus']['zero_media']:.3f}) | "
          f"full sigma share {g['fracao_sigma_completa_pct']['media']:.3f}% (projected {g['fracao_sigma_projetada_pct']['media']:.3f}%, zero {g['fracao_sigma_completa_pct']['zero_media']:.3f}%) | "
          f"outside span {g['energia_fora_do_span_pct']['media']:.1f}% | agreement whole {g['concordancia']['inteira']:.3f} sigma {g['concordancia']['sigma']:.3f} off-diagonal {g['concordancia']['fora']:.3f} spectrum {g['concordancia']['espectro']:.3f}")
    #endregion


if __name__ == "__main__":
    main()
