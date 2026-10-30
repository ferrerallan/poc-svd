"""
* what? - Aggregates one stage: means over the 10 seeds of the WEIGHT and DIRECTION percentages, plus accuracies.
* why?  - The page builders read one agregado.json per stage instead of re-reading 10 comparison files.
* how?  - Converts sigma variation and direction turn to percentages, averages per matrix and attaches reports.

Output: <stage folder>/agregado.json
Run from the repository root:  python3 doutorado/experimento1/common/aggregate_stage.py --stage 1|2
"""
import argparse
import json
import math
import statistics as st

import stages

K = math.pi / 180 * 100   # radians -> % (a unit vector turned by theta moves by theta)


def main():
    """
    * what? - Builds and writes agregado.json for the chosen stage, and cross-checks stage 1 against the old file.
    * why?  - It is the single table of results per stage: percentages, accuracies, agreement and decomposition.
    * how?  - Reads the 10 comparison files, the 10 training reports, the reproducibility and decomposition JSONs.
    """
    #region 1 - Parse flags and load the comparisons
    # what? - Reads the stage and indexes each seed's results by (block, type).
    # how?  - Seed 0 defines the list of matrices.
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, default=1)
    args = ap.parse_args()
    stage = args.stage

    results = {s: {(x["bloco"], x["tipo"]): x for x in json.load(open(stages.comparison(stage, s)))} for s in stages.SEEDS}
    keys = sorted(results[0])
    #endregion

    #region 2 - Average per matrix over the seeds
    # what? - One cell per matrix with the mean and standard deviation of both percentages.
    # how?  - p is the sigma change in % of the weights; d is the direction turn converted to %.
    cells = []
    for (b, t) in keys:
        p = [100 * results[s][(b, t)]["variacao_dos_valores"] for s in stages.SEEDS]
        d = [K * results[s][(b, t)]["giro_dos_vetores_graus"] for s in stages.SEEDS]
        cells.append({"b": b, "t": t, "p": round(st.mean(p), 5), "d": round(st.mean(d), 4),
                      "p_dp": round(st.stdev(p), 5), "d_dp": round(st.stdev(d), 4)})
    #endregion

    #region 3 - Attach reports and write the aggregate
    # what? - Adds accuracies, agreement and decomposition and saves agregado.json.
    # how?  - The keys stay in Portuguese: the page builders read this schema.
    reports = [json.load(open(stages.model_dir(stage, s) / "relatorio.json")) for s in stages.SEEDS]
    reproducibility = json.load(open(stages.REPRODUCIBILITY[stage]))
    decomposition_file = stages.STAGE_DIR[stage] / "decomposicao.json"
    out = {
        "etapa": stage, "weight_decay": stages.WEIGHT_DECAY[stage], "n": len(stages.SEEDS), "celulas": cells,
        "global": {"p": round(st.mean(c["p"] for c in cells), 5), "d": round(st.mean(c["d"] for c in cells), 4),
                   "cp": round(st.mean(x["cos_pesos"] for x in reproducibility), 4),
                   "cd": round(st.mean(x["cos_direcoes"] for x in reproducibility), 4),
                   "pos": sum(1 for x in reproducibility if x["cos_pesos"] > 0 and x["cos_direcoes"] > 0)},
        "treino": [{"semente": r["semente"], "antes": r["acuracia_antes"], "depois": r["acuracia_depois"],
                    "minutos": r["minutos_de_treino"]} for r in reports],
        "colados": sum(x["sigmas_colados_no_topo"] for x in results[0].values()), "pares_top": 10 * len(keys),
        "decomposicao": json.load(open(decomposition_file)) if decomposition_file.exists() else None,
    }
    (stages.STAGE_DIR[stage] / "agregado.json").write_text(json.dumps(out, indent=1))
    print(f"stage {stage}: weights {out['global']['p']:.4f}% | directions {out['global']['d']:.3f}% | "
          f"agreement weights {out['global']['cp']:.3f} directions {out['global']['cd']:.3f}")
    #endregion

    #region 4 - Cross-check stage 1 against the original aggregation
    # what? - Compares with resultado_10sementes.json, produced by analyze_seeds.py in stage 1.
    # how?  - Assertion with tolerance; a missing file is skipped silently.
    if stage == 1:
        try:
            old = json.load(open(stages.TEN_SEEDS_JSON))["global"]
            old_p = 100 * old["variacao_dos_valores"]["adaptacao"]["media"]
            old_d = K * old["giro_dos_vetores_graus"]["adaptacao"]["media"]
            assert abs(old_p - out["global"]["p"]) < 1e-4 and abs(old_d - out["global"]["d"]) < 1e-3, (old_p, old_d)
            print("checked: matches resultado_10sementes.json of stage 1")
        except FileNotFoundError:
            pass
    #endregion


if __name__ == "__main__":
    main()
