"""
* what? - Session 15 of the diary: does the footprint that FREE fine-tuning leaves on sigma (and on directions) point the same way as the solution found by the RESTRICTED trainings of stage 4?
* why?  - If it does, the SVD measurement of one free fine-tuning predicts the sigma adapter and is a cheap diagnostic; if it does not, that measurement says nothing about the sigma channel.
* how?  - Per matrix (72) and seed: free footprint on sigma = diag(U0^T dW V0) / S0 (384 numbers); sigma-only solution = delta from parametros_restritos.pt; cosine between them, against free-free and delta-delta agreement as references. Same for the off-diagonal part against the rotation-only change.

Models read (never modified): models/dinov2-small (W0), models/etapa3/s0..s9 (free), models/etapa4/so_sigma_s0..s9 (delta), models/etapa4/so_rotacao_s0..s9.
Run from the repository root:
    python3 doutorado/experimento1/etapa4/scripts/correlate_free_vs_restricted.py | tee doutorado/experimento1/etapa4/logs/correlacao_livre_vs_restrito.log
Output: doutorado/experimento1/etapa4/correlacao_livre_vs_restrito.json
"""
import itertools
import json
import statistics as st
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import Dinov2Model

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config_etapa4 as C  # noqa: E402

TOP = 10
OUTPUT = C.STAGE4 / "correlacao_livre_vs_restrito.json"


def cosine(a, b):
    """
    * what? - Cosine between two tensors of any shape.
    * why?  - It is the alignment measure used throughout the diary.
    * how?  - Flattened dot product over the product of norms.
    """
    #region 1 - Compute
    # what? - A float in [-1, 1].
    # how?  - 1e-12 guards against a zero vector.
    a, b = a.flatten(), b.flatten()
    return (a @ b / (a.norm() * b.norm() + 1e-12)).item()
    #endregion


def rank_correlation(x, y):
    """
    * what? - Spearman rank correlation between two lists.
    * why?  - Compares which matrices the free training and the restricted training touched most, without assuming a linear relation.
    * how?  - Ranks both lists and applies the Pearson formula to the ranks.
    """
    #region 1 - Rank and correlate
    # what? - A float in [-1, 1].
    # how?  - Average ranks are not needed: values are continuous and distinct.
    rx = torch.tensor(x).argsort().argsort().float()
    ry = torch.tensor(y).argsort().argsort().float()
    rx, ry = rx - rx.mean(), ry - ry.mean()
    return (rx @ ry / (rx.norm() * ry.norm())).item()
    #endregion


def main():
    """
    * what? - Computes the alignments and writes the JSON.
    * why?  - Gives session 15 its numbers.
    * how?  - Loads the 72 original matrices once, then the 30 trained models; one pass per matrix.
    """
    #region 1 - Load W0, the matrix names in the order of the restricted parameters, and the models
    # what? - names72 in named_parameters order (the order of parametros_restritos.pt), W0 per name, free dW, delta and rotation change per seed.
    # how?  - The order rule copies Classifier.__init__ of train_etapa4.py.
    body = Dinov2Model.from_pretrained(str(C.ORIGINAL_MODEL))
    names72 = [n for n, p in body.named_parameters() if p.dim() == 2 and n.startswith("encoder.layer.")]
    original = load_file(str(C.ORIGINAL_MODEL / "model.safetensors"))
    free = [load_file(str(C.full_model(s))) for s in C.SEEDS]
    delta = [torch.load(C.MODELS / f"so_sigma_s{s}" / "parametros_restritos.pt", map_location="cpu") for s in C.SEEDS]
    rot = [load_file(str(C.variant_model("so_rotacao", s))) for s in C.SEEDS]
    pairs_same = [(s, s) for s in C.SEEDS]
    pairs_diff = list(itertools.combinations(C.SEEDS, 2))
    pairs_cross = [(a, b) for a in C.SEEDS for b in C.SEEDS if a != b]
    #endregion

    #region 2 - Per matrix: footprints and alignments
    # what? - For each matrix: cosine free-sigma-footprint x delta (paired seeds and all pairs), free x free, delta x delta; same for off-diagonal x rotation change; magnitudes.
    # how?  - d/S0 is the relative stretch the free training applied to each original layer; delta is the stretch sigma-only learned.
    cells = []
    for i, n in enumerate(names72):
        W0 = original[n].float()
        U0, S0, Vt0 = torch.linalg.svd(W0, full_matrices=False)
        foot, off, dl, rch = [], [], [], []
        for s in C.SEEDS:
            dW = free[s][n].float() - W0
            d = ((U0.T @ dW) * Vt0).sum(1)
            foot.append(d / S0)
            off.append(dW - (U0 * d) @ Vt0)
            dl.append(delta[s][f"restrito_{i}"].float())
            rch.append(rot[s][n].float() - W0)
        cell = {"matrix": n, "shape": list(W0.shape),
                "cos_sigma_paired": st.mean(cosine(foot[a], dl[b]) for a, b in pairs_same),
                "cos_sigma_other_seeds": st.mean(cosine(foot[a], dl[b]) for a, b in pairs_cross),
                "cos_sigma_top10_paired": st.mean(cosine(foot[a][:TOP], dl[b][:TOP]) for a, b in pairs_same),
                "cos_sigma_top10_other_seeds": st.mean(cosine(foot[a][:TOP], dl[b][:TOP]) for a, b in pairs_cross),
                "cos_free_free_sigma_top10": st.mean(cosine(foot[a][:TOP], foot[b][:TOP]) for a, b in pairs_diff),
                "cos_delta_delta_top10": st.mean(cosine(dl[a][:TOP], dl[b][:TOP]) for a, b in pairs_diff),
                "cos_free_free_sigma": st.mean(cosine(foot[a], foot[b]) for a, b in pairs_diff),
                "cos_delta_delta": st.mean(cosine(dl[a], dl[b]) for a, b in pairs_diff),
                "sign_agreement_sigma": st.mean(((foot[a].sign() == dl[b].sign()).float().mean().item()) for a, b in pairs_same),
                "cos_direction_paired": st.mean(cosine(off[a], rch[b]) for a, b in pairs_same),
                "cos_direction_other_seeds": st.mean(cosine(off[a], rch[b]) for a, b in pairs_cross),
                "cos_free_free_direction": st.mean(cosine(off[a], off[b]) for a, b in pairs_diff),
                "cos_rot_rot": st.mean(cosine(rch[a], rch[b]) for a, b in pairs_diff),
                "free_sigma_mean_abs_pct": 100 * st.mean(f.abs().mean().item() for f in foot),
                "delta_mean_abs_pct": 100 * st.mean(x.abs().mean().item() for x in dl),
                "free_offdiag_rel": st.mean((o.norm() / W0.norm()).item() for o in off),
                "rot_change_rel": st.mean((r.norm() / W0.norm()).item() for r in rch)}
        cells.append(cell)
        print(f"{n:<48} cos sigma {cell['cos_sigma_paired']:+.3f} (free-free {cell['cos_free_free_sigma']:+.3f}, delta-delta {cell['cos_delta_delta']:+.3f})  "
              f"cos dir {cell['cos_direction_paired']:+.3f} (free-free {cell['cos_free_free_direction']:+.3f}, rot-rot {cell['cos_rot_rot']:+.3f})", flush=True)
    #endregion

    #region 3 - Summaries across the 72 matrices and write
    # what? - Means of every alignment (overall and per matrix type), the chance level of a cosine between random 384-vectors, and the rank correlations of magnitudes.
    # how?  - Chance level is 1/sqrt(384) as the standard deviation of a random cosine; rank correlation over the 72 per-matrix magnitudes; type is the last name before .weight.
    numeric = [k for k in cells[0] if k not in ("matrix", "shape")]
    summary = {k: st.mean(c[k] for c in cells) for k in numeric}
    by_type = {}
    for c in cells:
        kind = c["matrix"].split(".")[-2]
        by_type.setdefault(kind, []).append(c)
    summary["by_type"] = {kind: {k: st.mean(c[k] for c in group) for k in ("cos_sigma_paired", "cos_sigma_other_seeds", "cos_sigma_top10_paired", "cos_direction_paired", "free_sigma_mean_abs_pct", "delta_mean_abs_pct", "free_offdiag_rel", "rot_change_rel")}
                         for kind, group in by_type.items()}
    summary["chance_cosine_sd_384"] = 1 / 384 ** 0.5
    summary["spearman_magnitude_sigma"] = rank_correlation([c["free_sigma_mean_abs_pct"] for c in cells], [c["delta_mean_abs_pct"] for c in cells])
    summary["spearman_magnitude_direction"] = rank_correlation([c["free_offdiag_rel"] for c in cells], [c["rot_change_rel"] for c in cells])
    summary["matrices_cos_sigma_above_0_3"] = sum(c["cos_sigma_paired"] > 0.3 for c in cells)
    summary["matrices_cos_sigma_below_minus_0_3"] = sum(c["cos_sigma_paired"] < -0.3 for c in cells)
    OUTPUT.write_text(json.dumps({"top": TOP, "cells": cells, "summary": summary}, indent=1))
    print("\nSUMMARY (mean over 72 matrices)")
    for k, v in summary.items():
        if k == "by_type":
            for kind, vals in v.items():
                print(f"  {kind:>6}: " + "  ".join(f"{kk}={vv:.3f}" for kk, vv in vals.items()))
        else:
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    print("written:", OUTPUT.relative_to(C.ROOT))
    #endregion


if __name__ == "__main__":
    main()
