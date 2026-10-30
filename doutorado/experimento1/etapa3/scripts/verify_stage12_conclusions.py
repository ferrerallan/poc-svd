"""
* what? - STAGE 3, step 0: READ-ONLY checks over the models and JSONs of stages 1 and 2 that motivate stage 3.
* why?  - Tests the conclusions of stages 1-2 critically before building on them (several turned out wrong or weak).
* how?  - Seven independent checks over existing files; nothing from earlier stages is modified.

Output: etapa3/verificacao_etapas12.json
Checks:
  1. functional gain of body fine-tuning (fine-tuned x frozen body + logistic regression)
  2. directions/weights ratio in the noise between trainings
  3. turn x number of "stuck" sigma pairs per matrix
  4. vector-by-vector turn x principal angles of the top-10 SUBSPACE
  5. paired test (same seeds) of the weights, stage 1 x stage 2
  6. overall scale PER MATRIX in stage 2: sign consistency between trainings
  7. directions/weights ratio produced by ISOTROPIC noise without learning (the structural "zero")
Run from the repository root:  python3 doutorado/experimento1/etapa3/scripts/verify_stage12_conclusions.py
"""
import json
import math
import statistics as st
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[4]            # repository root
EXPERIMENT = ROOT / "doutorado/experimento1"
sys.path.insert(0, str(EXPERIMENT / "common"))
import stages  # noqa: E402  (read-only use)

K = math.pi / 180 * 100
TOP = 10


def load_json(path):
    """
    * what? - Reads a JSON file into Python objects.
    * why?  - The checks read many JSON files.
    * how?  - json.load on an opened file.
    """
    #region 1 - Read the file
    # what? - Parses the JSON.
    # how?  - The handle is closed by the garbage collector.
    return json.load(open(path))
    #endregion


def pearson(a, b):
    """
    * what? - Pearson correlation coefficient of two equally long lists.
    * why?  - Check 3 correlates the turn with the number of stuck sigma pairs.
    * how?  - Covariance over the product of the standard deviations, written out with sums.
    """
    #region 1 - Compute the coefficient
    # what? - r in [-1, 1].
    # how?  - Centered sums of products and squares.
    ma, mb = st.mean(a), st.mean(b)
    return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / math.sqrt(sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b))
    #endregion


def per_seed(stage):
    """
    * what? - Returns the mean weight change in % of each seed of a stage.
    * why?  - Check 5 pairs stage 1 and stage 2 seed by seed.
    * how?  - Mean of variacao_dos_valores over the matrices of each comparison JSON, times 100.
    """
    #region 1 - Collect the per-seed means
    # what? - One value per seed.
    # how?  - Reads the comparison JSON of the stage and seed.
    return [100 * st.mean(x["variacao_dos_valores"] for x in load_json(ROOT / stages.comparison(stage, s))) for s in stages.SEEDS]
    #endregion


def main():
    """
    * what? - Runs the seven checks and writes verificacao_etapas12.json.
    * why?  - The result decided what stage 3 had to correct: ruler, functional test and control.
    * how?  - Each check fills one key of the output dictionary; the isotropic-noise check seeds torch with 0.
    """
    #region 1 - Load the original model
    # what? - Original weights and the sorted list of the 72 matrix names.
    # how?  - Sorted by block number, then by name.
    original = load_file(ROOT / "models/dinov2-small/model.safetensors")
    names = sorted([n for n, t in original.items() if t.dim() == 2 and "encoder.layer." in n], key=lambda n: (int(n.split(".")[2]), n))
    out = {}
    #endregion

    #region 2 - Check 1: functional gain
    # what? - Compares the fine-tuned accuracies with the frozen-body accuracy.
    # how?  - Frozen-body value comes from etapa1/diagnostico.json.
    diag = load_json(EXPERIMENT / "etapa1/diagnostico.json")["acuracia_corpo_congelado"]
    out["funcional"] = {"corpo_congelado_reg_logistica": diag}
    for e in (1, 2):
        acc = [t["depois"] for t in load_json(EXPERIMENT / f"etapa{e}/agregado.json")["treino"]]
        out["funcional"][f"etapa{e}"] = {"media": st.mean(acc), "dp": st.stdev(acc), "ganho_sobre_congelado": st.mean(acc) - diag}
    #endregion

    #region 3 - Check 2: directions/weights ratio in the noise
    # what? - Ratio of direction % to weight % for adaptation and for noise, stage 1.
    # how?  - K converts degrees to %.
    g = load_json(ROOT / stages.TEN_SEEDS_JSON)["global"]
    dir_noise, weights_noise = K * g["giro_dos_vetores_graus"]["ruido"]["media"], 100 * g["variacao_dos_valores"]["ruido"]["media"]
    dir_adapt, weights_adapt = K * g["giro_dos_vetores_graus"]["adaptacao"]["media"], 100 * g["variacao_dos_valores"]["adaptacao"]["media"]
    out["razao_direcoes_pesos"] = {"adaptacao_etapa1": dir_adapt / weights_adapt, "ruido_etapa1": dir_noise / weights_noise}
    #endregion

    #region 4 - Check 3: turn versus stuck sigmas
    # what? - Correlation between the number of stuck sigmas and the vector-by-vector turn.
    # how?  - Compares the groups with up to 3 stuck sigmas and with more than 3.
    cells = load_json(ROOT / stages.TEN_SEEDS_JSON)["celulas"]
    stuck = {(x["bloco"], x["tipo"]): x["sigmas_colados_no_topo"] for x in load_json(ROOT / stages.comparison(1, 0))}
    pairs = [(stuck[(c["bloco"], c["tipo"])], c["giro_dos_vetores_graus"]["adaptacao"]["media"]) for c in cells]
    out["giro_x_colados"] = {"r_pearson": pearson([p[0] for p in pairs], [p[1] for p in pairs]),
                             "giro_ate_3_colados": st.mean(y for x, y in pairs if x <= 3), "n_ate_3": sum(1 for x, _ in pairs if x <= 3),
                             "giro_mais_de_3": st.mean(y for x, y in pairs if x > 3), "n_mais_de_3": sum(1 for x, _ in pairs if x > 3)}
    #endregion

    #region 5 - Check 4: vector-by-vector turn versus subspace angles
    # what? - Compares the two ways of measuring the turn on stage 2 seed 0.
    # how?  - Principal angles come from the singular values of U0' U1.
    m0 = load_file(ROOT / stages.model(2, 0))
    individual, sub_mean, sub_max = [], [], []
    for n in names:
        U0 = torch.linalg.svd(original[n].float(), full_matrices=False)[0][:, :TOP]
        U1 = torch.linalg.svd(m0[n].float(), full_matrices=False)[0][:, :TOP]
        individual.append(torch.rad2deg(torch.arccos((U0 * U1).sum(0).abs().clamp(max=1))).mean().item())
        angles = torch.rad2deg(torch.arccos(torch.linalg.svdvals(U0.T @ U1).clamp(max=1)))
        sub_mean.append(angles.mean().item())
        sub_max.append(angles.max().item())
    out["subespaco_top10_etapa2_s0"] = {"giro_vetor_a_vetor_graus": st.mean(individual), "angulo_principal_medio_graus": st.mean(sub_mean),
                                        "angulo_principal_maior_graus": st.mean(sub_max), "fracao_subespaco_sobre_vetores": st.mean(sub_mean) / st.mean(individual)}
    #endregion

    #region 6 - Check 5: paired test of the weights, stage 1 versus stage 2
    # what? - Paired differences by seed.
    # how?  - t statistic of the mean difference over its standard error.
    p1, p2 = per_seed(1), per_seed(2)
    diff = [a - b for a, b in zip(p1, p2)]
    out["pareado_pesos_e1_e2"] = {"media_diferenca_pct": st.mean(diff), "dp": st.stdev(diff), "t_pareado": st.mean(diff) / (st.stdev(diff) / math.sqrt(len(diff))),
                                  "positivo_em": sum(x > 0 for x in diff), "n": len(diff)}
    #endregion

    #region 7 - Check 6: overall scale per matrix in stage 2
    # what? - In how many matrices the overall sigma scale has the same sign in 9 or 10 trainings.
    # how?  - Compared with the number expected by chance from a binomial with p = 0.5.
    models2 = {s: load_file(ROOT / stages.model(2, s)) for s in stages.SEEDS}
    consistency, magnitude = [], []
    for n in names:
        s0 = torch.linalg.svdvals(original[n].float())[:TOP]
        signs, values = [], []
        for s in stages.SEEDS:
            c = (((torch.linalg.svdvals(models2[s][n].float())[:TOP] - s0) / s0).mean()).item()
            signs.append(c > 0)
            values.append(c)
        consistency.append(max(sum(signs), 10 - sum(signs)))
        magnitude.append(abs(100 * st.mean(values)))
    by_chance_9 = 72 * 2 * (math.comb(10, 9) + math.comb(10, 10)) / 2 ** 10
    out["escala_geral_por_matriz_etapa2"] = {"matrizes_com_sinal_igual_em_9_ou_10_treinos": sum(c >= 9 for c in consistency), "esperado_por_acaso": by_chance_9,
                                            "matrizes_com_8_ou_mais": sum(c >= 8 for c in consistency), "modulo_medio_pct": st.mean(magnitude)}
    #endregion

    #region 8 - Check 7: isotropic noise without learning
    # what? - Direction/weight ratio produced by random noise of 0.5% of the matrix norm.
    # how?  - torch.manual_seed(0) right before the loop keeps the noise reproducible.
    torch.manual_seed(0)
    ratios = []
    for n in names:
        W = original[n].float()
        U0, S0, _ = torch.linalg.svd(W, full_matrices=False)
        noise = torch.randn_like(W)
        noise *= 0.005 * W.norm() / noise.norm()
        U1, S1, _ = torch.linalg.svd(W + noise, full_matrices=False)
        weights_pct = 100 * ((S1[:TOP] - S0[:TOP]).abs() / S0[:TOP]).mean().item()
        directions_pct = 100 * torch.arccos((U0[:, :TOP] * U1[:, :TOP]).sum(0).abs().clamp(max=1)).mean().item()
        ratios.append(directions_pct / weights_pct)
    out["zero_isotropico_0p5pct"] = {"razao_mediana": st.median(ratios), "razao_min": min(ratios), "razao_max": max(ratios), "n_matrizes": len(ratios)}
    #endregion

    #region 9 - Write the result
    # what? - Saves verificacao_etapas12.json and prints it.
    # how?  - The keys stay in Portuguese: the stage 3 page builder reads this schema.
    (EXPERIMENT / "etapa3/verificacao_etapas12.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    #endregion


if __name__ == "__main__":
    main()
