"""
* what? - Study test of session 9: how the body stays frozen in epoch 1 and is released from epoch 2 in common/train.py.
* why?  - Backs the description in the diary with measurements instead of reading the code and assuming.
* how?  - Rebuilds the model, optimizer and schedule exactly as train.py does, runs epoch 1 frozen and 10 batches of epoch 2 released, and compares the body with the original after each phase.

Measured: gradient of the body (none or not), body tensors changed, how much the 72 matrices changed, and the step size (learning rate) of each group.
Run from the repository root:
    python3 doutorado/experimento1/etapa1/scripts/freeze_release_check.py | tee doutorado/experimento1/etapa1/estudos/congelar_soltar.log
Output: doutorado/experimento1/etapa1/estudos/congelar_soltar.json
"""
import json
import statistics as st
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "doutorado/experimento1/common"))
import train as T  # noqa: E402

OUTPUT = Path(__file__).resolve().parents[1] / "estudos" / "congelar_soltar.json"
RELEASED_BATCHES = 10
EPOCHS = T.EPOCHS


def snapshot(model):
    """
    * what? - Returns a CPU copy of every body tensor.
    * why?  - The comparison needs the weights as they were at a given moment.
    * how?  - detach().clone() of each named parameter, in float32.
    """
    #region 1 - Copy the body
    # what? - Dictionary name -> tensor.
    # how?  - The body has 223 parameter tensors (the same 223 as in the safetensors file).
    return {name: p.detach().float().cpu().clone() for name, p in model.body.named_parameters()}
    #endregion


def compare(before, model):
    """
    * what? - Counts the body tensors that changed and measures the change of the 72 big matrices.
    * why?  - Frozen means zero changes; released means many small ones.
    * how?  - torch.equal per tensor; for the matrices, the norm of the difference divided by the norm of the original.
    """
    #region 1 - Compare every tensor
    # what? - Number changed, total, and the relative change of the 72 matrices.
    # how?  - The 72 matrices are the 2-D weights inside encoder.layer (query, key, value, dense, fc1, fc2 of 12 blocks).
    after = snapshot(model)
    unchanged = [n for n in before if torch.equal(before[n], after[n])]
    changed = len(before) - len(unchanged)
    matrices = [n for n in before if n.startswith("encoder.layer") and n.endswith(".weight") and before[n].dim() == 2]
    assert len(matrices) == 72, len(matrices)
    relative = [((after[n] - before[n]).norm() / before[n].norm()).item() for n in matrices]
    return {"tensors": len(before), "changed": changed, "unchanged_names": unchanged, "matrix_relative_change_median": st.median(relative),
            "matrix_relative_change_max": max(relative)}
    #endregion


def schedule_table(total_steps):
    """
    * what? - Returns the step size (learning rate) of the body and of the head at chosen steps of the 575.
    * why?  - Shows that the schedule keeps running while the body is frozen, so the body is released with a step already below 1e-5.
    * how?  - Steps a dummy optimizer with the same two groups and the same cosine schedule, recording the values before each step.
    """
    #region 1 - Run the schedule on a dummy
    # what? - Dictionary step -> (body lr, head lr).
    # how?  - Step k uses the value held before schedule.step() is called for that step.
    dummy = nn.Parameter(torch.zeros(1))
    opt = torch.optim.AdamW([{"params": [dummy], "lr": T.BODY_LR}, {"params": [nn.Parameter(torch.zeros(1))], "lr": T.HEAD_LR}])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, total_steps)
    lrs = []
    for _ in range(total_steps):
        lrs.append((opt.param_groups[0]["lr"], opt.param_groups[1]["lr"]))
        opt.step()
        sched.step()
    return lrs
    #endregion


def main():
    """
    * what? - Runs the check and prints what happened in each phase.
    * why?  - Gives session 9 its measured description of freezing and releasing.
    * how?  - Same setup as train.py main(): two parameter groups, AdamW 0.05, cosine schedule over 5 epochs, label smoothing 0.1, bf16.
    """
    #region 1 - Build the same setup as train.py
    # what? - Data, model, optimizer, schedule and loss.
    # how?  - Seed 0, batch 32, drop_last so an epoch has 115 batches.
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train, _ = T.build_data()
    loader = DataLoader(train, batch_size=T.BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    model = T.PetClassifier(T.MODEL_BEFORE, T.NUM_CLASSES).to(device)
    optimizer = torch.optim.AdamW([{"params": model.body.parameters(), "lr": T.BODY_LR},
                                   {"params": model.head.parameters(), "lr": T.HEAD_LR}], weight_decay=0.05)
    total_steps = EPOCHS * len(loader)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_steps)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    original = snapshot(model)
    head_before = model.head.weight.detach().clone()
    lrs = schedule_table(total_steps)
    print(f"total steps {total_steps} ({len(loader)} per epoch); body tensors {len(original)}")
    #endregion

    #region 2 - Epoch 1: body frozen
    # what? - 115 steps with requires_grad False on every body weight.
    # how?  - The gradient stops at the head; the body keeps grad None, which AdamW skips (weight decay included).
    for p in model.body.parameters():
        p.requires_grad_(False)
    model.train()
    seen = []
    for step, (photos, labels) in enumerate(loader, start=1):
        seen.append((optimizer.param_groups[0]["lr"], optimizer.param_groups[1]["lr"]))
        with torch.autocast(device, dtype=torch.bfloat16):
            loss = loss_fn(model(photos.to(device)), labels.to(device))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        schedule.step()
    none_grads = sum(1 for p in model.body.parameters() if p.grad is None)
    frozen = compare(original, model)
    head_change = ((model.head.weight.detach() - head_before).norm() / head_before.norm()).item()
    assert seen[0] == lrs[0] and seen[-1] == lrs[len(loader) - 1], "schedule differs from the dummy"
    print(f"\nEPOCH 1 (frozen), {len(loader)} steps")
    print(f"  body tensors with grad None: {none_grads} of {len(original)}")
    print(f"  body tensors changed: {frozen['changed']} of {frozen['tensors']}")
    print(f"  head weight change (norm of difference / norm of start): {head_change:.2f}")
    #endregion

    #region 3 - Epoch 2: body released
    # what? - The first batches after requires_grad is set to True again.
    # how?  - Nothing else is reset: same optimizer, same schedule position, Adam starts with no history for the body.
    for p in model.body.parameters():
        p.requires_grad_(True)
    first_lr = (optimizer.param_groups[0]["lr"], optimizer.param_groups[1]["lr"])
    for step, (photos, labels) in enumerate(loader, start=1):
        if step > RELEASED_BATCHES:
            break
        with torch.autocast(device, dtype=torch.bfloat16):
            loss = loss_fn(model(photos.to(device)), labels.to(device))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        schedule.step()
        if step == 1:
            first_released = compare(original, model)
    released = compare(original, model)
    print(f"\nEPOCH 2 (released), first {RELEASED_BATCHES} steps")
    print(f"  step size at release (step {len(loader) + 1}): body {first_lr[0]:.3e}, head {first_lr[1]:.3e}")
    print(f"  after 1 step: body tensors changed {first_released['changed']} of {first_released['tensors']}")
    print(f"  tensors still unchanged: {released['unchanged_names']} (never used when the model classifies photos, so it gets no gradient)")
    print(f"  after {RELEASED_BATCHES} steps: changed {released['changed']} of {released['tensors']}; "
          f"median change of the 72 matrices {released['matrix_relative_change_median']:.2e}, largest {released['matrix_relative_change_max']:.2e}")
    #endregion

    #region 4 - The whole schedule and the JSON
    # what? - Step size of both groups at chosen steps of the 575.
    # how?  - Taken from the dummy run, which the real run matched at the steps it covered.
    marks = [1, len(loader), len(loader) + 1, 345, total_steps]
    table = {str(k): {"body_lr": lrs[k - 1][0], "head_lr": lrs[k - 1][1]} for k in marks}
    print("\nSTEP SIZE along the 575 steps (cosine schedule, same for frozen and released)")
    for k in marks:
        print(f"  step {k:>3}: body {lrs[k - 1][0]:.3e}   head {lrs[k - 1][1]:.3e}")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps({"steps_per_epoch": len(loader), "total_steps": total_steps, "frozen_epoch": {**frozen, "body_grad_none": none_grads, "head_relative_change": head_change},
                                  "released_first_step": first_released, "released_after_batches": {"batches": RELEASED_BATCHES, **released},
                                  "step_size_at_release": {"body": first_lr[0], "head": first_lr[1]}, "step_size_table": table}, indent=2))
    print("\nwritten:", OUTPUT.relative_to(ROOT))
    #endregion


if __name__ == "__main__":
    main()
