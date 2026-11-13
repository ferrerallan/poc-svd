"""
* what? - STAGE 11 (session 22, checkpoint): re-reads the summaries of the five model and domain pairs and checks, by code, every claim that the diary keeps as valid for all of them.
* why?  - After 21 sessions some claims fell and some hold; the checkpoint must list only the ones that the stored numbers support in every pair, and a script makes that list reproducible instead of a reading from memory.
* how?  - Loads the five resumo JSONs (stages 5, 7, 8, 9, 10) and the two GPU cost JSONs (stages 4 and 8), builds one row per pair, evaluates each claim as a boolean over the rows and writes everything to etapa11/afirmacoes_etapa11.json.

Nothing is trained here. Output: etapa11/afirmacoes_etapa11.json
Run from the repository root:  python3 doutorado/experimento1/etapa11/scripts/check_claims_etapa11.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
E1 = ROOT / "doutorado/experimento1"
OUT = E1 / "etapa11/afirmacoes_etapa11.json"

# (pair label, summary file, key of the sigma-only variant, model, domain, numbers in the 72 matrices)
PAIRS = [
    ("pele_small", "etapa5/resumo_etapa5.json", "so_sigma_passo_novo", "dinov2-small", "pele", 21233664),
    ("kmnist_small", "etapa7/resumo_etapa7.json", "so_sigma", "dinov2-small", "kmnist", 21233664),
    ("pele_base", "etapa8/resumo_etapa8.json", "so_sigma", "dinov2-base", "pele", 84934656),
    ("dtd_small", "etapa9/resumo_etapa9.json", "so_sigma", "dinov2-small", "dtd", 21233664),
    ("eurosat_small", "etapa10/resumo_etapa10.json", "so_sigma", "dinov2-small", "eurosat", 21233664),
]
GPU = [("dinov2-small", "etapa4/custo_gpu.json"), ("dinov2-base", "etapa8/custo_gpu_base.json")]
TRIAGE = {"pele_small": 56.58, "kmnist_small": 79.34, "pele_base": 59.15, "dtd_small": 78.62, "eurosat_small": 94.88}
BIG_GAIN = 10.0  # points; the three pairs above it are the "large gain" group of session 21


def mean(xs):
    """
    * what? - Arithmetic mean of a list.
    * why?  - Paired differences between variants are averaged over the 10 seeds.
    * how?  - Sum divided by length.
    """
    #region 1 - Compute
    # what? - The mean as a float.
    # how?  - Plain Python, no numpy needed.
    return sum(xs) / len(xs)
    #endregion


def spearman(xs, ys):
    """
    * what? - Spearman rank correlation between two lists without ties.
    * why?  - Tests whether the frozen-body screening orders the pairs the same way as the gain or the recovered fraction.
    * how?  - Ranks both lists and applies 1 - 6 sum(d^2) / (n (n^2 - 1)).
    """
    #region 1 - Rank and combine
    # what? - A float between -1 and 1.
    # how?  - Ranks are positions in the sorted list, 1-based.
    rx = {v: i + 1 for i, v in enumerate(sorted(xs))}
    ry = {v: i + 1 for i, v in enumerate(sorted(ys))}
    n = len(xs)
    d2 = sum((rx[a] - ry[b]) ** 2 for a, b in zip(xs, ys))
    return 1 - 6 * d2 / (n * (n * n - 1))
    #endregion


def pair_row(label, summary, sigma_key, model, domain, n72):
    """
    * what? - Builds one row of numbers for a model and domain pair from its stage summary.
    * why?  - Every claim is evaluated over these rows, so they hold everything the claims need.
    * how?  - Reads accuracies, per-seed lists, displacement, top-10 sigma change, preserved directions and chosen steps of the four variants.
    """
    #region 1 - Load the summary and the four variants
    # what? - The dicts of frozen 72, sigma only, 3-block rotation and full training.
    # how?  - Stage 5 keeps the 3e-2 sigma run under so_sigma_passo_novo; the others keep it under so_sigma.
    d = json.load(open(E1 / summary))
    cong, sig, rot, full = d["congelado_72"], d[sigma_key], d["so_rotacao_bloco3"], d["completo"]
    acc = lambda v: v["acuracia"]["media"]
    seeds = lambda v: v["acuracia"]["por_semente"]
    gain = acc(full) - acc(cong)
    #endregion
    #region 2 - Paired differences over the seeds
    # what? - full minus sigma and sigma minus rotation, mean and number of seeds with the first above the second.
    # how?  - Same seed index on both lists; the lists are already ordered s0 to s9.
    fs = [f - s for f, s in zip(seeds(full), seeds(sig))]
    sr = [s - r for s, r in zip(seeds(sig), seeds(rot))]
    #endregion
    #region 3 - Assemble the row
    # what? - A flat dict with every number used by the claims and by the tables of session 22.
    # how?  - Percentages are kept as in the summaries (0 to 100); fractions of the gain are in percent too.
    return {
        "pair": label, "model": model, "domain": domain, "triage_pct": TRIAGE[label],
        "acc_frozen72": acc(cong), "acc_sigma": acc(sig), "acc_rotation3": acc(rot), "acc_full": acc(full),
        "gain_points": gain,
        "sigma_recovers_pct": 100 * (acc(sig) - acc(cong)) / gain,
        "rotation_recovers_pct": 100 * (acc(rot) - acc(cong)) / gain,
        "full_minus_sigma_mean": mean(fs), "full_minus_sigma_seeds_above": sum(1 for x in fs if x > 0),
        "sigma_minus_rotation_mean": mean(sr), "sigma_minus_rotation_seeds_above": sum(1 for x in sr if x > 0),
        "sigma_params": sig["params_restritos_efetivos"], "rotation_params": rot["params_restritos_efetivos"],
        "numbers_in_72": n72, "sigma_params_pct_of_72": 100 * sig["params_restritos_efetivos"] / n72,
        "lr_sigma": sig["lr_restrito"], "lr_rotation": rot["lr_restrito"],
        "disp_sigma_pct": sig["desloc_pct"], "disp_rotation_pct": rot["desloc_pct"], "disp_full_pct": full["desloc_pct"],
        "sigma_top10_sigma_pct": sig["sigma_top10_abs_pct"], "sigma_top10_rotation_pct": rot["sigma_top10_abs_pct"],
        "sigma_top10_full_pct": full["sigma_top10_abs_pct"],
        "directions_preserved_sigma": sig["direcoes_preservadas"], "directions_preserved_rotation": rot["direcoes_preservadas"],
        "directions_preserved_full": full["direcoes_preservadas"],
    }
    #endregion


def gpu_rows():
    """
    * what? - Time per step and gradient plus optimizer memory of sigma only against full training, in both models.
    * why?  - The computational claim must be checked in the two models, not quoted from the text of sessions 14 and 19.
    * how?  - Reads the rows of the two cost JSONs and keeps the so_sigma and completo variants.
    """
    #region 1 - Read both cost files
    # what? - One dict per model with the ratios used by the claim.
    # how?  - Memory of the trainable numbers is gradients plus optimizer state; peak memory is reported as measured.
    rows = []
    for model, path in GPU:
        d = json.load(open(E1 / path))
        by = {r["variant"]: r for r in d["rows"]}
        s, f = by["so_sigma"], by["completo"]
        rows.append({
            "model": model, "gpu": d["gpu"], "batch_size": d["batch_size"],
            "seconds_per_step_sigma": s["seconds_per_training_step"], "seconds_per_step_full": f["seconds_per_training_step"],
            "time_ratio_sigma_over_full": s["seconds_per_training_step"] / f["seconds_per_training_step"],
            "grad_opt_mb_sigma": s["gradients_mb"] + s["optimizer_state_mb"], "grad_opt_mb_full": f["gradients_mb"] + f["optimizer_state_mb"],
            "grad_opt_ratio_full_over_sigma": (f["gradients_mb"] + f["optimizer_state_mb"]) / (s["gradients_mb"] + s["optimizer_state_mb"]),
            "peak_mb_sigma": s["peak_training_mb"], "peak_mb_full": f["peak_training_mb"],
            "peak_saving_pct": 100 * (1 - s["peak_training_mb"] / f["peak_training_mb"]),
        })
    return rows
    #endregion


def claims(rows, gpu):
    """
    * what? - Evaluates the nine claims of the checkpoint over the five pair rows and the two GPU rows.
    * why?  - A claim enters session 22 only if its boolean is true here.
    * how?  - Each claim is a dict with an id, a one-line statement, the numbers that support it and a boolean holds.
    """
    #region 1 - Groups of pairs
    # what? - Large-gain and small-gain pairs, by the threshold of session 21.
    # how?  - Gain above BIG_GAIN points is large.
    big = [r for r in rows if r["gain_points"] > BIG_GAIN]
    small = [r for r in rows if r["gain_points"] <= BIG_GAIN]
    out = []
    #endregion
    #region 2 - Claims about accuracy
    # what? - Sigma recovers most of the gain; full stays above sigma; at equal cost sigma never loses by much and wins clearly when the gain is large.
    # how?  - Min and max over the rows, plus seed counts.
    frac = [r["sigma_recovers_pct"] for r in rows]
    out.append({"id": 1, "statement": "Sigma only recovers most of the gain of the 72 matrices in every pair, with at most 0.13% of their numbers.",
                "min_recovered_pct": min(frac), "max_recovered_pct": max(frac), "max_sigma_params_pct_of_72": max(r["sigma_params_pct_of_72"] for r in rows),
                "holds": min(frac) >= 75.0})
    fms = [r["full_minus_sigma_mean"] for r in rows]
    out.append({"id": 2, "statement": "Full training stays above sigma only in every pair.",
                "min_gap_points": min(fms), "max_gap_points": max(fms), "min_seeds_above": min(r["full_minus_sigma_seeds_above"] for r in rows),
                "holds": min(fms) > 0 and min(r["full_minus_sigma_seeds_above"] for r in rows) >= 8})
    out.append({"id": 3, "statement": "At comparable cost, sigma only never lost to the 3-block rotation by more than 0.3 point, and in the large-gain pairs it won by 1.3 to 2.3 points in 10 of 10 seeds.",
                "worst_sigma_minus_rotation": min(r["sigma_minus_rotation_mean"] for r in rows),
                "large_gain_pairs": [r["pair"] for r in big], "large_gain_min_margin": min(r["sigma_minus_rotation_mean"] for r in big),
                "large_gain_max_margin": max(r["sigma_minus_rotation_mean"] for r in big), "large_gain_min_seeds": min(r["sigma_minus_rotation_seeds_above"] for r in big),
                "small_gain_pairs": [r["pair"] for r in small], "small_gain_max_abs_margin": max(abs(r["sigma_minus_rotation_mean"]) for r in small),
                "holds": min(r["sigma_minus_rotation_mean"] for r in rows) >= -0.3 and all(r["sigma_minus_rotation_mean"] >= 1.3 and r["sigma_minus_rotation_seeds_above"] == 10 for r in big)})
    #endregion
    #region 3 - Claims about what moved inside the matrices
    # what? - The restrictions held; free training barely touches sigma while sigma only moves it a lot; displacement is set by the protocol.
    # how?  - Bands over the five rows.
    out.append({"id": 4, "statement": "The restrictions held in every pair: sigma only kept the directions and the rotation kept the sigmas.",
                "min_directions_preserved_sigma": min(r["directions_preserved_sigma"] for r in rows),
                "max_sigma_change_rotation_pct": max(r["sigma_top10_rotation_pct"] for r in rows),
                "holds": min(r["directions_preserved_sigma"] for r in rows) >= 0.9999 and max(r["sigma_top10_rotation_pct"] for r in rows) <= 1e-4})
    ratio = [r["sigma_top10_sigma_pct"] / r["sigma_top10_full_pct"] for r in rows]
    out.append({"id": 5, "statement": "Free training changes the 10 largest sigmas by about 0.02% in every pair, while sigma only needs 3% to 12% to do its share of the work.",
                "full_sigma_change_min_pct": min(r["sigma_top10_full_pct"] for r in rows), "full_sigma_change_max_pct": max(r["sigma_top10_full_pct"] for r in rows),
                "sigma_only_change_min_pct": min(r["sigma_top10_sigma_pct"] for r in rows), "sigma_only_change_max_pct": max(r["sigma_top10_sigma_pct"] for r in rows),
                "ratio_min": min(ratio), "ratio_max": max(ratio),
                "holds": max(r["sigma_top10_full_pct"] for r in rows) <= 0.02 and min(r["sigma_top10_sigma_pct"] for r in rows) >= 3.0})
    s1 = [r["disp_sigma_pct"] for r in rows if r["lr_sigma"] == 0.01]
    s3 = [r["disp_sigma_pct"] for r in rows if r["lr_sigma"] == 0.03]
    rt = [r["disp_rotation_pct"] for r in rows]
    fl = [r["disp_full_pct"] for r in rows]
    out.append({"id": 6, "statement": "The displacement of the 72 matrices is set by the variant and the step, not by the domain or the model.",
                "sigma_lr1e-2_pct": [min(s1), max(s1)], "sigma_lr3e-2_pct": [min(s3), max(s3)], "rotation_pct": [min(rt), max(rt)], "full_pct": [min(fl), max(fl)],
                "holds": max(s1) - min(s1) <= 1.0 and max(s3) - min(s3) <= 1.5 and max(rt) - min(rt) <= 1.0 and max(fl) - min(fl) <= 0.2})
    #endregion
    #region 4 - Claims about the protocol and the cost
    # what? - Steps, GPU time and memory, and the screening.
    # how?  - Equalities over the rows and ratios over the GPU rows.
    by_dom = {}
    for r in rows:
        by_dom.setdefault(r["domain"], set()).add(r["lr_sigma"])
    out.append({"id": 7, "statement": "The rotation step was 1e-3 in every pair; the sigma step depended on the domain and not on the model.",
                "rotation_steps": sorted({r["lr_rotation"] for r in rows}), "sigma_step_by_domain": {k: sorted(v) for k, v in by_dom.items()},
                "holds": {r["lr_rotation"] for r in rows} == {0.001} and all(len(v) == 1 for v in by_dom.values()) and len({r["lr_sigma"] for r in rows}) > 1})
    out.append({"id": 8, "statement": "Sigma only saves no time per step and saves most of the gradient and optimizer memory, in both models.",
                "gpu": gpu, "holds": all(abs(g["time_ratio_sigma_over_full"] - 1) <= 0.05 and g["grad_opt_ratio_full_over_sigma"] >= 20 for g in gpu)})
    tri = [r["triage_pct"] for r in rows]
    rho_gain = spearman(tri, [r["gain_points"] for r in rows])
    rho_frac = spearman(tri, [r["sigma_recovers_pct"] for r in rows])
    out.append({"id": 9, "statement": "The frozen-body screening predicts neither the total gain nor the fraction that sigma only recovers.",
                "spearman_triage_vs_gain": rho_gain, "spearman_triage_vs_sigma_fraction": rho_frac, "n_pairs": len(rows),
                "holds": abs(rho_gain) < 0.5 and abs(rho_frac) < 0.5})
    return out
    #endregion


def main():
    """
    * what? - Builds the rows, evaluates the claims and writes the JSON; exits 1 if any claim fails.
    * why?  - Session 22 is generated from this file, so a failing claim must be visible before the diary is written.
    * how?  - Prints one line per claim with its boolean.
    """
    #region 1 - Rows and claims
    # what? - Five pair rows, two GPU rows, nine claims.
    # how?  - Pure reads; nothing on the GPU.
    rows = [pair_row(*p) for p in PAIRS]
    gpu = gpu_rows()
    cl = claims(rows, gpu)
    #endregion
    #region 2 - Write and report
    # what? - afirmacoes_etapa11.json and one line per claim on stdout.
    # how?  - Exit code 1 if a claim does not hold, so the recipe stops.
    OUT.write_text(json.dumps({"pairs": rows, "gpu": gpu, "claims": cl}, indent=1, ensure_ascii=False), encoding="utf-8")
    for c in cl:
        print(f"claim {c['id']}: {'holds' if c['holds'] else 'FAILS'} | {c['statement']}")
    print("written:", OUT.relative_to(ROOT))
    sys.exit(0 if all(c["holds"] for c in cl) else 1)
    #endregion


if __name__ == "__main__":
    main()
