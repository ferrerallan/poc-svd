"""
* what? - Summary of the 10 head-only trainings on Oxford-IIIT Pet: accuracy per seed and per epoch, and proof that the body did not change.
* why?  - Session 8 of the diary talks only about training the head; this gives it a measured result with its spread.
* how?  - Reads relatorio.json of each model, compares the 223 body tensors with the original model, and counts the head numbers.

The trainings come from train.py with the body frozen for all 5 epochs (see reproduce_head_only.sh).
Run from the repository root:
    python3 doutorado/experimento1/etapa1/scripts/head_only_summary.py | tee doutorado/experimento1/etapa1/estudos/so_cabeca.log
Output: doutorado/experimento1/etapa1/estudos/so_cabeca.json
"""
import json
import statistics as st
from pathlib import Path

import torch
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[4]
ORIGINAL = ROOT / "models/dinov2-small/model.safetensors"
TRAINED = ROOT / "models/etapa1_so_cabeca"
OUTPUT = Path(__file__).resolve().parents[1] / "estudos" / "so_cabeca.json"
SEEDS = list(range(10))


def body_difference(trained_file):
    """
    * what? - Returns how many of the 223 body tensors differ from the original model and the largest absolute difference.
    * why?  - If the body is frozen, both must be zero; this checks it instead of assuming it.
    * how?  - Opens both safetensors files and compares tensor by tensor.
    """
    #region 1 - Compare tensor by tensor
    # what? - Count of tensors that changed and the largest difference found.
    # how?  - torch.equal for the count, (a - b).abs().max() for the size of the difference.
    changed, largest = 0, 0.0
    with safe_open(str(ORIGINAL), "pt") as a, safe_open(str(trained_file), "pt") as b:
        names = list(a.keys())
        for name in names:
            x, y = a.get_tensor(name), b.get_tensor(name)
            if not torch.equal(x, y):
                changed += 1
                largest = max(largest, (x.float() - y.float()).abs().max().item())
    return len(names), changed, largest
    #endregion


def main():
    """
    * what? - Prints a table with one line per seed and writes the JSON summary.
    * why?  - Gives session 8 its numbers: accuracy, spread and the check that the body stayed the same.
    * how?  - Loops over seeds 0 to 9, then takes mean, minimum and maximum of the final accuracy.
    """
    #region 1 - Read every seed
    # what? - Accuracy before, per epoch and at the end, plus the body check, for each seed.
    # how?  - relatorio.json is written by train.py; the head is cabeca.pt (weight 37 x 384 and bias 37).
    rows = []
    print(f"{'seed':>4}  {'before':>7}  {'epoch1':>7}  {'epoch2':>7}  {'epoch5':>7}  {'tensors':>8}  {'changed':>8}  {'largest diff':>12}")
    for seed in SEEDS:
        folder = TRAINED / f"s{seed}"
        report = json.loads((folder / "relatorio.json").read_text())
        total, changed, largest = body_difference(folder / "model.safetensors")
        per_epoch = report["acuracias_por_epoca"]
        rows.append({"seed": seed, "accuracy_before": report["acuracia_antes"], "accuracy_per_epoch": per_epoch,
                     "accuracy_after": report["acuracia_depois"], "body_tensors": total,
                     "body_tensors_changed": changed, "body_largest_difference": largest})
        print(f"{seed:>4}  {report['acuracia_antes']:>7.2f}  {per_epoch[0]:>7.2f}  {per_epoch[1]:>7.2f}  {per_epoch[-1]:>7.2f}  {total:>8}  {changed:>8}  {largest:>12.1e}")
    #endregion

    #region 2 - Count the head and summarize
    # what? - Number of head weights and the statistics over the seeds.
    # how?  - The head shape is read from the first seed's cabeca.pt.
    head = torch.load(TRAINED / "s0" / "cabeca.pt", map_location="cpu")
    head_weights, head_bias = head["weight"].numel(), head["bias"].numel()
    final = [r["accuracy_after"] for r in rows]
    epoch1 = [r["accuracy_per_epoch"][0] for r in rows]
    summary = {"accuracy_after": {"mean": st.mean(final), "std": st.stdev(final), "min": min(final), "max": max(final)},
               "accuracy_epoch1_mean": st.mean(epoch1), "head_weights": head_weights, "head_bias": head_bias,
               "body_tensors_changed_total": sum(r["body_tensors_changed"] for r in rows)}
    print(f"\nfinal accuracy: mean {summary['accuracy_after']['mean']:.2f}  std {summary['accuracy_after']['std']:.2f}  min {min(final):.2f}  max {max(final):.2f}")
    print(f"head: {head_weights} weights + {head_bias} bias;  body tensors changed in all seeds: {summary['body_tensors_changed_total']}")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps({"seeds": SEEDS, "rows": rows, "summary": summary}, indent=2))
    print("written:", OUTPUT.relative_to(ROOT))
    #endregion


if __name__ == "__main__":
    main()
