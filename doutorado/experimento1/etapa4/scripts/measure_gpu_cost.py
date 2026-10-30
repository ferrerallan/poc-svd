"""
* what? - STAGE 4, session 14 of the diary: measures the GPU cost of training each variant (frozen 72, sigma only, rotation only, full).
* why?  - Restricting the trainable numbers is supposed to save something; this measures what, instead of assuming (peak memory, optimizer state, gradients, time per step).
* how?  - Builds each variant exactly as train_etapa4.py does, runs STEPS training steps on batches preloaded to the GPU (so data loading does not enter the timing), and reads torch.cuda memory counters.

Measured per variant: trainable numbers; bytes of weights, of gradients and of the AdamW state; peak GPU memory during a training step
(body free, as in epochs 2 to 5); seconds per training step; seconds per evaluation batch of 64 photos (forward only).
Run from the repository root:
    python3 doutorado/experimento1/etapa4/scripts/measure_gpu_cost.py | tee doutorado/experimento1/etapa4/logs/custo_gpu.log
Output: doutorado/experimento1/etapa4/custo_gpu.json
"""
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "etapa3" / "scripts"))
import config_etapa4 as C  # noqa: E402
import train_etapa3 as T3  # noqa: E402
import train_etapa4 as T4  # noqa: E402

STEPS, WARMUP, EVAL_BATCHES = 30, 5, 20
RESTRICTED_LR = {"so_sigma": 1e-2, "so_rotacao": 1e-4}          # the steps chosen in passos_escolhidos.json
OUTPUT = C.STAGE4 / "custo_gpu.json"
MB = 1024 ** 2


def tensor_bytes(tensors):
    """
    * what? - Sums the bytes of a list of tensors.
    * why?  - Weights, gradients and optimizer states are all lists of tensors.
    * how?  - numel times element size, skipping None.
    """
    #region 1 - Sum
    # what? - Total bytes.
    # how?  - element_size() gives 4 for float32 and 2 for bfloat16.
    return sum(t.numel() * t.element_size() for t in tensors if t is not None)
    #endregion


def preload_batches(loader, n, device):
    """
    * what? - Copies the first n batches of a loader to the GPU.
    * why?  - Timing must measure the model, not the JPEG decoding of the data loader.
    * how?  - Iterates the loader once and stops at n.
    """
    #region 1 - Copy
    # what? - List of (photos, labels) pairs on the device.
    # how?  - non_blocking is irrelevant here; the copy is done once.
    out = []
    for x, y in loader:
        out.append((x.to(device), y.to(device)))
        if len(out) == n:
            break
    return out
    #endregion


def measure_variant(variant, train_batches, eval_batches, n_classes, device, data_mb):
    """
    * what? - Returns the cost dictionary of one variant.
    * why?  - One function, applied to the four variants, keeps the measurement identical.
    * how?  - Build, count, train STEPS steps with timing after WARMUP, read peak memory (minus the preloaded photos), then time the forward pass in eval mode.
    """
    #region 1 - Build the model and the optimizer as train_etapa4.py does
    # what? - Classifier, parameter groups and AdamW; memory after the build is the weight footprint.
    # how?  - Body and restricted parameters are set free (epochs 2 to 5 regime) before the first step.
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    torch.manual_seed(0)
    model = T4.Classifier(variant, n_classes, C.OFT_BLOCK, "exp").to(device)
    restricted, body, head = model.groups()
    groups = [{"params": body, "lr": T4.BODY_LR}, {"params": head, "lr": T4.HEAD_LR}]
    if restricted:
        groups.insert(0, {"params": restricted, "lr": RESTRICTED_LR[variant]})
    optimizer = torch.optim.AdamW(groups, weight_decay=C.WEIGHT_DECAY)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    for p in body + restricted:
        p.requires_grad_(True)
    torch.cuda.synchronize(device)
    weights_mb = torch.cuda.memory_allocated(device) / MB - data_mb
    trainable = [p for p in model.parameters() if p.requires_grad]
    frozen = [p for p in model.parameters() if not p.requires_grad]
    #endregion

    #region 2 - Training steps, timed after the warm-up
    # what? - Seconds per step and peak memory of the training regime.
    # how?  - cuda.synchronize before reading the clock; the first WARMUP steps are not timed (allocator and kernels warm up).
    model.train()
    times = []
    for i in range(STEPS):
        x, y = train_batches[i % len(train_batches)]
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = loss_fn(model(x), y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize(device)
        if i >= WARMUP:
            times.append(time.perf_counter() - t0)
    peak_mb = torch.cuda.max_memory_allocated(device) / MB - data_mb
    grad_mb = tensor_bytes([p.grad for p in trainable]) / MB
    state_mb = tensor_bytes([v for s in optimizer.state.values() for v in s.values() if torch.is_tensor(v)]) / MB
    #endregion

    #region 3 - Evaluation forward pass, timed
    # what? - Seconds per batch of 64 photos without gradients.
    # how?  - Same autocast as the training evaluation; first batch excluded.
    model.eval()
    eval_times = []
    with torch.no_grad():
        for i, (x, _) in enumerate(eval_batches):
            torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                model(x)
            torch.cuda.synchronize(device)
            if i > 0:
                eval_times.append(time.perf_counter() - t0)
    #endregion

    #region 4 - Collect
    # what? - The dictionary of this variant.
    # how?  - Trainable numbers split as train_etapa4.py reports them; bytes in MB (1024 x 1024); memory figures exclude the preloaded photos.
    result = {"variant": variant, "trainable_numbers": sum(p.numel() for p in trainable), "frozen_numbers": sum(p.numel() for p in frozen),
              "trainable_in_72_matrices": sum(p.numel() for p in restricted) if restricted else (sum(p.numel() for n, p in model.body.named_parameters() if n in model.names72 and p.requires_grad)),
              "weights_mb": round(weights_mb, 1), "gradients_mb": round(grad_mb, 1), "optimizer_state_mb": round(state_mb, 1),
              "peak_training_mb": round(peak_mb, 1), "seconds_per_training_step": round(sum(times) / len(times), 4),
              "seconds_per_eval_batch_64": round(sum(eval_times) / len(eval_times), 4)}
    del model, optimizer, trainable, frozen, restricted, body, head
    torch.cuda.empty_cache()
    return result
    #endregion


def main():
    """
    * what? - Measures the four variants and writes the JSON.
    * why?  - Gives session 14 its numbers.
    * how?  - Same skin data and batch size as the trainings; batches preloaded once and shared by all variants.
    """
    #region 1 - Data
    # what? - 8 training batches of 32 and 21 evaluation batches of 64, on the GPU, and the memory they take.
    # how?  - Same namespace as train_etapa4.py main().
    device = "cuda"
    import types
    ns = types.SimpleNamespace(train=str(C.SKIN_TRAIN), test=str(C.SKIN_TEST), image_col="image", label_col="label",
                               max_train=3680, max_test=3669, shuffle_labels=False)
    train, test, classes = T3.build_data(ns)
    train_batches = preload_batches(DataLoader(train, batch_size=T4.BATCH_SIZE, shuffle=True, num_workers=4, drop_last=True), 8, device)
    eval_batches = preload_batches(DataLoader(test, batch_size=64, shuffle=False, num_workers=4), EVAL_BATCHES + 1, device)
    torch.cuda.synchronize(device)
    data_mb = torch.cuda.memory_allocated(device) / MB
    print(f"gpu: {torch.cuda.get_device_name(0)} | batch {T4.BATCH_SIZE} | {STEPS} steps per variant ({WARMUP} warm-up) | preloaded photos {data_mb:.0f} MB (excluded from the memory figures)")
    #endregion

    #region 2 - Measure every variant and write
    # what? - One row per variant, printed and saved.
    # how?  - Order: frozen 72, sigma only, rotation only, full.
    rows = []
    print(f"{'variant':>13} {'trainable':>11} {'weights MB':>10} {'grads MB':>9} {'adam MB':>8} {'peak MB':>8} {'s/step':>7} {'s/eval64':>8}")
    for v in ("congelado_72", "so_sigma", "so_rotacao", "completo"):
        r = measure_variant(v, train_batches, eval_batches, len(classes), device, data_mb)
        rows.append(r)
        print(f"{v:>13} {r['trainable_numbers']:>11,} {r['weights_mb']:>10} {r['gradients_mb']:>9} {r['optimizer_state_mb']:>8} {r['peak_training_mb']:>8} {r['seconds_per_training_step']:>7} {r['seconds_per_eval_batch_64']:>8}")
    OUTPUT.write_text(json.dumps({"gpu": torch.cuda.get_device_name(0), "batch_size": T4.BATCH_SIZE, "steps": STEPS, "warmup_steps": WARMUP,
                                  "torch": torch.__version__, "preloaded_photos_mb": round(data_mb, 1), "rows": rows}, indent=1))
    print("written:", OUTPUT.relative_to(C.ROOT))
    #endregion


if __name__ == "__main__":
    main()
