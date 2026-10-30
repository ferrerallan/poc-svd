"""
* what? - STAGE 4: RESTRICTED fine-tuning of DINOv2-Small on skin lesions, same protocol as stage 3, changing only the FREEDOM of the 72 matrices.
* why?  - Tests which kind of freedom (sigma only, rotation only) is enough to learn, instead of only measuring free training.
* how?  - Replaces the 72 linear layers by sigma-only or rotation-only modules (or freezes them), then trains like stage 3.

  completo      W free                                   (reference: the 10 stage 3 models; this script can also train it)
  congelado_72  the 72 matrices stay at W0; everything else (bias, LayerNorm, embeddings, head) trains as in completo
  so_sigma      W = U0 diag(sigma0 * (1 + delta)) V0^T, trains only delta (r numbers per matrix)      -> SVF / SVDiff style
  so_rotacao    W = R W0, R block-diagonal orthogonal (from a skew-symmetric matrix), trains only R   -> OFT style
                (an orthogonal R preserves ALL singular values of W0; only the U directions change)
In every variant the remaining body parameters and the head train exactly as in stage 3 (step 1e-5 / 1e-3, wd 0,
5 epochs, 1 frozen-body warm-up epoch, batch 32, cosine, label smoothing 0.1, bf16).
The step of the restricted parameters (delta or R) is --restricted-lr, chosen in the pilots by VALIDATION (--validation 0.1).
The saved model is a standard Dinov2Model (W materialized), so the same measuring instruments work.
Reuses, unchanged, the data loading of etapa3/scripts/train_etapa3.py.
Run from the repository root:  python3 doutorado/experimento1/etapa4/scripts/train_etapa4.py --variant so_sigma --restricted-lr 1e-3 --output <folder>
"""
import argparse
import datetime
import hashlib
import json
import platform
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from torch.utils.data import DataLoader, Subset
from transformers import Dinov2Model

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "etapa3" / "scripts"))
import config_etapa4 as C  # noqa: E402
import train_etapa3 as T3  # noqa: E402  (read-only use: build_data, accuracy)

EPOCHS, WARMUP_EPOCHS, BATCH_SIZE, BODY_LR, HEAD_LR = 5, 1, 32, 1e-5, 1e-3


class LinearSigma(nn.Module):
    """
    * what? - Linear layer W = U0 diag(sigma0 * (1 + delta)) V0^T where only delta trains; U0 and V0 are fixed.
    * why?  - Isolates the "weights" half of the thesis question: can learning happen by re-weighting directions alone?
    * how?  - SVD of the original W0 stored as buffers; delta is a zero-initialized parameter; materialize() rebuilds W on the fly.
    """

    def __init__(self, W0, bias):
        """
        * what? - Decomposes W0 and creates the zero-initialized delta and the trainable bias.
        * why?  - At delta = 0 the layer equals the original one exactly.
        * how?  - U0, S0 and Vt0 are buffers (not trained); delta has the shape of S0.
        """
        #region 1 - Decompose W0 and create the parameters
        # what? - Buffers for the SVD factors and the parameters delta and bias.
        # how?  - The bias is copied so it can train independently of the original layer.
        super().__init__()
        U, S, Vt = torch.linalg.svd(W0.float(), full_matrices=False)
        self.register_buffer("U0", U)
        self.register_buffer("S0", S)
        self.register_buffer("Vt0", Vt)
        self.delta = nn.Parameter(torch.zeros_like(S))
        self.bias = nn.Parameter(bias.detach().clone()) if bias is not None else None
        #endregion

    def materialize(self):
        """
        * what? - Rebuilds the weight matrix from the fixed directions and the re-weighted sigmas.
        * why?  - The matrix is never stored; gradients flow into delta only.
        * how?  - (U0 * (S0 * (1 + delta))) @ Vt0.
        """
        #region 1 - Rebuild W
        # what? - Returns the matrix.
        # how?  - Column scaling of U0 by the new sigmas, then the product with Vt0.
        return (self.U0 * (self.S0 * (1 + self.delta))) @ self.Vt0
        #endregion

    def forward(self, x):
        """
        * what? - Applies the linear layer with the rebuilt weight.
        * why?  - Drop-in replacement for nn.Linear inside the transformer.
        * how?  - F.linear with the weight cast to the input dtype (bf16 autocast).
        """
        #region 1 - Apply the layer
        # what? - Returns W x + b.
        # how?  - The bias is cast like the weight.
        return F.linear(x, self.materialize().to(x.dtype), None if self.bias is None else self.bias.to(x.dtype))
        #endregion


class LinearRotation(nn.Module):
    """
    * what? - Linear layer W = R W0 with R block-diagonal orthogonal; only the skew-symmetric generators train.
    * why?  - Isolates the "directions" half of the question: R preserves every singular value of W0.
    * how?  - R_b = exp(2A_b) (or Cayley (I+A)(I-A)^-1) per block, A = B - B^T; materialize() applies R to the row blocks of W0.
    """

    def __init__(self, W0, bias, block, parametrization="exp"):
        """
        * what? - Stores W0 and creates the zero-initialized generators B for every row block.
        * why?  - At B = 0, R is the identity and the layer equals the original one.
        * how?  - The number of rows must be divisible by the block size; B has shape (n_blocks, block, block).
        """
        #region 1 - Create the buffers and parameters
        # what? - W0 buffer, generators B and the bias.
        # how?  - assert guards the divisibility of the row count by the block size.
        super().__init__()
        m = W0.shape[0]
        assert m % block == 0, (m, block)
        self.parametrization = parametrization
        self.register_buffer("W0", W0.float().clone())
        self.n_blocks, self.block_size = m // block, block
        self.B = nn.Parameter(torch.zeros(self.n_blocks, block, block))        # A = B - B^T
        self.bias = nn.Parameter(bias.detach().clone()) if bias is not None else None
        #endregion

    def rotation(self):
        """
        * what? - Builds the block-diagonal orthogonal matrices R from the generators.
        * why?  - R must be exactly orthogonal so the singular values of W0 are preserved.
        * how?  - A = B - B^T is skew-symmetric; exp(2A) (default) or the Cayley transform give an orthogonal R.
        """
        #region 1 - Build R
        # what? - Returns the stack of block rotations.
        # how?  - Cayley was used in the first pilots; its CUDA solver failed in a long training, so exp is the default.
        A = self.B - self.B.transpose(1, 2)
        if self.parametrization == "cayley":
            I = torch.eye(self.block_size, device=A.device, dtype=A.dtype).expand_as(A)
            return torch.linalg.solve(I - A, I + A)
        return torch.linalg.matrix_exp(2 * A)                               # exp(2A): orthogonal, matches Cayley up to 2nd order in A
        #endregion

    def materialize(self):
        """
        * what? - Returns R W0 with R applied block by block to the row blocks of W0.
        * why?  - The rotated matrix is never stored; gradients flow into B only.
        * how?  - W0 viewed as (n_blocks, block, columns) and a batched matrix product with R.
        """
        #region 1 - Apply the rotation
        # what? - Returns the matrix with the shape of W0.
        # how?  - bmm over the blocks, then view back to the original shape.
        W = self.W0.view(self.n_blocks, self.block_size, -1)
        return torch.bmm(self.rotation(), W).view_as(self.W0)
        #endregion

    def forward(self, x):
        """
        * what? - Applies the linear layer with the rotated weight.
        * why?  - Drop-in replacement for nn.Linear inside the transformer.
        * how?  - F.linear with the weight cast to the input dtype (bf16 autocast).
        """
        #region 1 - Apply the layer
        # what? - Returns W x + b.
        # how?  - The bias is cast like the weight.
        return F.linear(x, self.materialize().to(x.dtype), None if self.bias is None else self.bias.to(x.dtype))
        #endregion


class Classifier(nn.Module):
    """
    * what? - DINOv2 body whose 72 matrices follow the chosen variant, plus a new linear head on the CLS token.
    * why?  - One model class serves all variants so the training loop is identical.
    * how?  - Replaces each of the 72 linear layers by LinearSigma or LinearRotation, or freezes them, depending on the variant.
    """

    def __init__(self, variant, num_classes, block, rotation_param="exp"):
        """
        * what? - Builds body and head and applies the variant to the 72 matrices.
        * why?  - Restrictions are implemented by swapping the layers, not by masking gradients.
        * how?  - Walks the names of the 72 matrices, finds each parent module and replaces or freezes the layer.
        """
        #region 1 - Build body, head and the list of the 72 matrices
        # what? - Pre-trained body, random head and the matrix names.
        # how?  - Body first, then the head draws its random numbers: the order matters for reproducibility.
        super().__init__()
        self.body = Dinov2Model.from_pretrained(str(C.ORIGINAL_MODEL))
        self.head = nn.Linear(self.body.config.hidden_size, num_classes)
        self.variant = variant
        self.names72 = [n for n, p in self.body.named_parameters() if p.dim() == 2 and n.startswith("encoder.layer.")]
        self.restricted = []
        #endregion

        #region 2 - Apply the variant to every matrix
        # what? - Replaces or freezes each of the 72 layers.
        # how?  - parent_name is e.g. encoder.layer.0.attention.attention.query; its parent holds the attribute to replace.
        for n in self.names72:
            parent_name, _ = n.rsplit(".", 1)
            parent = self.body.get_submodule(parent_name.rsplit(".", 1)[0])
            attr = parent_name.rsplit(".", 1)[1]
            lin = getattr(parent, attr)
            if variant == "so_sigma":
                new = LinearSigma(lin.weight.data, lin.bias.data if lin.bias is not None else None)
            elif variant == "so_rotacao":
                new = LinearRotation(lin.weight.data, lin.bias.data if lin.bias is not None else None, block, rotation_param)
            else:
                if variant == "congelado_72":
                    lin.weight.requires_grad_(False)
                continue
            setattr(parent, attr, new)
            self.restricted.append(new)
        #endregion

    def forward(self, pixels):
        """
        * what? - Maps a batch of photos to one score per class.
        * why?  - Classification reads only the CLS token.
        * how?  - Body, token 0 of the last hidden state, head.
        """
        #region 1 - Run body and head
        # what? - Returns the logits.
        # how?  - last_hidden_state[:, 0] is the CLS token.
        return self.head(self.body(pixel_values=pixels).last_hidden_state[:, 0])
        #endregion

    def groups(self):
        """
        * what? - Splits the trainable parameters into restricted ones (delta / B), the rest of the body and the head.
        * why?  - The three groups get different learning rates.
        * how?  - Collects delta and B from the replaced layers and filters them out of the body parameters by id.
        """
        #region 1 - Split the parameters
        # what? - Three lists.
        # how?  - Body parameters that do not require grad (frozen matrices) are left out.
        restricted = [m.delta for m in self.restricted if isinstance(m, LinearSigma)] + [m.B for m in self.restricted if isinstance(m, LinearRotation)]
        ids = {id(p) for p in restricted}
        body = [p for p in self.body.parameters() if id(p) not in ids and p.requires_grad]
        return restricted, body, list(self.head.parameters())
        #endregion

    def materialized_state_dict(self):
        """
        * what? - Returns the state dict of a standard Dinov2Model with the 72 matrices materialized.
        * why?  - The saved model must be loadable and measurable like every other model of the experiment.
        * how?  - Starts from the original keys; restricted layers contribute materialize(), everything else the trained tensor.
        """
        #region 1 - Materialize the matrices
        # what? - Builds the dictionary key by key.
        # how?  - Tensors are moved to CPU and cast to the dtype of the original checkpoint.
        base = Dinov2Model.from_pretrained(str(C.ORIGINAL_MODEL)).state_dict()
        current = {k: v.detach().cpu() for k, v in self.body.state_dict().items()}
        sd = {}
        for k in base:
            if k in self.names72 and self.variant in ("so_sigma", "so_rotacao"):
                mod = self.body.get_submodule(k.rsplit(".", 1)[0])
                sd[k] = mod.materialize().detach().cpu().to(base[k].dtype)
            else:
                sd[k] = current[k].to(base[k].dtype)
        return sd
        #endregion


def main():
    """
    * what? - Trains one variant and seed (or a step pilot) and writes the model, the head, the restricted parameters and the report.
    * why?  - Produces the stage 4 models: pilots for the step choice and the 10 seeds of each restricted variant.
    * how?  - Parse flags, build data (optionally with a validation split), build the classifier, train, materialize and save.
    """
    #region 1 - Parse flags and seed
    # what? - Reads the command line and fixes the seed.
    # how?  - The restricted variants require --restricted-lr.
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=["completo"] + C.VARIANTS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", required=True)
    ap.add_argument("--restricted-lr", type=float, default=None, dest="restricted_lr", help="step of delta (so_sigma) or of R (so_rotacao)")
    ap.add_argument("--validation", type=float, default=0.0, help="fraction of the train set kept for validation (step pilots)")
    ap.add_argument("--oft-block", type=int, default=C.OFT_BLOCK, dest="oft_block")
    ap.add_argument("--rotation-param", default="exp", choices=["exp", "cayley"], dest="rotation_param")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--warmup-epochs", type=int, default=WARMUP_EPOCHS, dest="warmup_epochs")
    a = ap.parse_args()
    if a.variant in ("so_sigma", "so_rotacao") and a.restricted_lr is None:
        ap.error("--restricted-lr is required for so_sigma and so_rotacao")
    started_iso = datetime.datetime.now().isoformat(timespec="seconds")
    torch.manual_seed(a.seed)
    device = "cuda"
    #endregion

    #region 2 - Prepare the data
    # what? - Skin-lesion datasets, with an optional validation split taken from the TRAIN set.
    # how?  - In the pilots the test set is not used; the validation split uses the fixed seed 4242.
    ns = types.SimpleNamespace(train=str(C.SKIN_TRAIN), test=str(C.SKIN_TEST), image_col="image", label_col="label",
                               max_train=3680, max_test=3669, shuffle_labels=False)
    train, test, classes = T3.build_data(ns)
    eval_name = "teste"
    if a.validation > 0:
        g = np.random.default_rng(4242)
        idx = g.permutation(len(train))
        n_val = int(round(a.validation * len(train)))
        test = Subset(train, idx[:n_val].tolist())
        train = Subset(train, idx[n_val:].tolist())
        eval_name = "validacao"
    train_loader = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
    print(f"1) variant {a.variant} | train {len(train)} | {eval_name} {len(test)} | classes {len(classes)} | restricted lr {a.restricted_lr}")
    #endregion

    #region 3 - Build the model and the optimizer
    # what? - Classifier of the variant, parameter groups, optimizer, schedule and loss.
    # how?  - The restricted group (if any) is inserted first, with its own learning rate.
    model = Classifier(a.variant, len(classes), a.oft_block, a.rotation_param).to(device)
    restricted, body, head = model.groups()
    n_restricted = sum(p.numel() for p in restricted)
    n_body = sum(p.numel() for p in body)
    groups = [{"params": body, "lr": BODY_LR}, {"params": head, "lr": HEAD_LR}]
    if restricted:
        groups.insert(0, {"params": restricted, "lr": a.restricted_lr})
    optimizer = torch.optim.AdamW(groups, weight_decay=C.WEIGHT_DECAY)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, a.epochs * len(train_loader))
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    print(f"   trainable parameters: restricted {n_restricted:,} | rest of the body {n_body:,} | head {sum(p.numel() for p in head):,}")
    acc0 = T3.accuracy(model, test_loader, device)
    print(f"   accuracy before training ({eval_name}): {acc0:.1f}%")
    #endregion

    #region 4 - Train
    # what? - Epoch loop with the frozen-body warm-up; records accuracy, train accuracy and loss per epoch.
    # how?  - During the warm-up the body and restricted parameters are frozen; they unfreeze afterwards.
    start = time.time()
    acc_epoch, train_acc_epoch, loss_epoch = [], [], []
    for epoch in range(1, a.epochs + 1):
        free = epoch > a.warmup_epochs
        for p in body + restricted:
            p.requires_grad_(free)
        model.train()
        loss_sum = 0.0
        hits = total = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = model(x)
                loss = loss_fn(output, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            schedule.step()
            loss_sum += loss.item()
            hits += (output.argmax(1) == y).sum().item()
            total += y.numel()
        acc = T3.accuracy(model, test_loader, device)
        acc_epoch.append(round(acc, 2))
        train_acc_epoch.append(round(100 * hits / total, 2))
        loss_epoch.append(round(loss_sum / len(train_loader), 4))
        print(f"   epoch {epoch}/{a.epochs}  {'free   ' if free else 'frozen '}  loss {loss_sum/len(train_loader):.3f}  train acc {100*hits/total:.1f}%  acc {eval_name} {acc:.1f}%", flush=True)
    minutes = (time.time() - start) / 60
    final_acc = T3.accuracy(model, test_loader, device)
    #endregion

    #region 5 - Save the model, the head and the restricted parameters
    # what? - Writes a standard Dinov2Model, the head and, when present, the restricted parameters.
    # how?  - The materialized state dict is loaded into a fresh Dinov2Model and saved with save_pretrained.
    dest = Path(a.output)
    dest.mkdir(parents=True, exist_ok=True)
    sd = model.materialized_state_dict()
    base = Dinov2Model.from_pretrained(str(C.ORIGINAL_MODEL))
    base.load_state_dict(sd)
    base.save_pretrained(dest)
    torch.save(model.head.state_dict(), dest / "cabeca.pt")
    if restricted:
        torch.save({f"restrito_{i}": p.detach().cpu() for i, p in enumerate(restricted)}, dest / "parametros_restritos.pt")
    #endregion

    #region 6 - Write the report
    # what? - relatorio.json with every setting and result of the run.
    # how?  - The keys stay in Portuguese: measure_variants.py and the page builder read this schema.
    report = {"etapa": 4, "variante": a.variant, "dataset": "pele", "treino_parquet": str(C.SKIN_TRAIN), "teste_parquet": str(C.SKIN_TEST),
              "n_treino": len(train), "n_avaliacao": len(test), "avaliacao": eval_name, "fracao_validacao": a.validation, "num_classes": len(classes),
              "semente": a.seed, "lr_restrito": a.restricted_lr, "bloco_oft": a.oft_block if a.variant == "so_rotacao" else None, "param_rotacao": a.rotation_param if a.variant == "so_rotacao" else None,
              "params_restritos": n_restricted, "params_restritos_efetivos": (sum(m.n_blocks * m.block_size * (m.block_size - 1) // 2 for m in model.restricted) if a.variant == "so_rotacao" else n_restricted), "params_resto_corpo": n_body, "params_cabeca": sum(p.numel() for p in head),
              "epocas": a.epochs, "epocas_aquecer": a.warmup_epochs, "lote": BATCH_SIZE, "lr_corpo": BODY_LR, "lr_cabeca": HEAD_LR, "weight_decay": C.WEIGHT_DECAY,
              "label_smoothing": 0.1, "otimizador": "AdamW", "agenda_lr": "cosseno, um passo por lote", "precisao": "bf16 (autocast)",
              "acuracia_antes": round(acc0, 2), "acuracia_depois": round(final_acc, 2), "acuracias_por_epoca": acc_epoch, "acuracia_treino_por_epoca": train_acc_epoch,
              "perda_por_epoca": loss_epoch, "minutos_de_treino": round(minutes, 1), "comando": " ".join(sys.argv), "iniciado_em": started_iso,
              "python": platform.python_version(), "torch": torch.__version__, "transformers": transformers.__version__, "gpu": torch.cuda.get_device_name(0),
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (dest / "relatorio.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nDONE in {minutes:.1f} min | {a.variant} | {eval_name}: {acc0:.1f}% -> {final_acc:.1f}% | saved in {a.output}/")
    #endregion


if __name__ == "__main__":
    main()
