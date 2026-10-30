"""
* what? - The two hand examples of sessions 8 and 9 of the diary: one gradient step on the head, and one gradient step on a body matrix.
* why?  - The diary quotes these numbers (0.974, 0.634, 0.440 and 0.936); this script reproduces them and checks them against PyTorch's automatic gradient.
* how?  - Computes each step by hand formula (softmax, probability minus target, outer product) and by torch.autograd, and prints both.

Example 1 (session 8): 2 classes, CLS = [1, 2], head rows A = [0.5, 0] and B = [0, 0.5], correct class A, step 0.1.
Example 2 (session 9): the same head stays fixed; the CLS now comes from a body matrix W = [[1, 0], [0, 2]] applied to the input x = [1, 1].
Run from the repository root:
    python3 doutorado/experimento1/etapa1/scripts/gradient_examples.py | tee doutorado/experimento1/etapa1/estudos/exemplos_gradiente.log
Output: doutorado/experimento1/etapa1/estudos/exemplos_gradiente.json
"""
import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[4]
OUTPUT = Path(__file__).resolve().parents[1] / "estudos" / "exemplos_gradiente.json"
STEP = 0.1


def loss_of(head, cls):
    """
    * what? - Returns the error (cross entropy) of the head on a CLS vector whose correct class is A (index 0).
    * why?  - It is the number the diary follows from step to step.
    * how?  - scores = head @ cls, then cross_entropy, which is softmax followed by minus the log of the correct probability.
    """
    #region 1 - Score and error
    # what? - A scalar tensor.
    # how?  - cross_entropy expects a batch, so the scores get a batch dimension of 1.
    scores = head @ cls
    return torch.nn.functional.cross_entropy(scores.unsqueeze(0), torch.tensor([0]))
    #endregion


def example_head():
    """
    * what? - Trains the head for two steps and returns the numbers quoted in session 8.
    * why?  - Shows how the label reaches the head weights.
    * how?  - Gradient by hand (probability minus target, times CLS) and by autograd; both must agree.
    """
    #region 1 - First step, by hand and by autograd
    # what? - Scores, probabilities, error, gradient and the new head.
    # how?  - By hand the gradient row of each class is (probability - target) * CLS.
    cls = torch.tensor([1.0, 2.0])
    head = torch.tensor([[0.5, 0.0], [0.0, 0.5]], requires_grad=True)
    scores = (head @ cls).detach()
    probs = torch.softmax(scores, 0)
    by_hand = torch.outer(probs - torch.tensor([1.0, 0.0]), cls)
    loss0 = loss_of(head, cls)
    loss0.backward()
    new_head = (head - STEP * head.grad).detach()
    agree = torch.allclose(by_hand, head.grad, atol=1e-6)
    #endregion

    #region 2 - Second step
    # what? - The error after one and after two steps.
    # how?  - The same computation repeated from the new head.
    h1 = new_head.clone().requires_grad_(True)
    loss1 = loss_of(h1, cls)
    loss1.backward()
    h2 = (h1 - STEP * h1.grad).detach()
    loss2 = loss_of(h2, cls)
    #endregion

    #region 3 - Collect and print
    # what? - A dictionary with the numbers of the diary table.
    # how?  - Tensors become lists so the JSON can hold them.
    result = {"scores": scores.tolist(), "probabilities": probs.tolist(), "loss_0": loss0.item(),
              "gradient": by_hand.tolist(), "new_head": new_head.tolist(), "gradient_by_hand_equals_autograd": agree,
              "loss_1": loss1.item(), "loss_2": loss2.item()}
    print("EXAMPLE 1 (session 8): one step on the head")
    print(f"  scores {[round(v, 4) for v in result['scores']]}  probabilities {[round(v, 4) for v in result['probabilities']]}")
    print(f"  loss {loss0.item():.3f}; gradient rows {[[round(v, 4) for v in r] for r in result['gradient']]}")
    print(f"  new head {[[round(v, 4) for v in r] for r in result['new_head']]}")
    print(f"  loss after step 1: {loss1.item():.3f}; after step 2: {loss2.item():.3f}; hand gradient == autograd: {agree}")
    return result
    #endregion


def example_body():
    """
    * what? - Applies one gradient step to a body matrix W while the head stays fixed, and returns the numbers quoted in session 9.
    * why?  - Shows how the label reaches the body: the error travels back through the head to the matrix.
    * how?  - Gradient by hand (head transposed times probability minus target, outer product with the input) and by autograd.
    """
    #region 1 - Gradient on W
    # what? - The error, the gradient of W and the new W.
    # how?  - dL/dCLS = head^T (p - y); dL/dW = dL/dCLS outer x, because CLS = W x.
    head = torch.tensor([[0.5, 0.0], [0.0, 0.5]])
    x = torch.tensor([1.0, 1.0])
    w = torch.tensor([[1.0, 0.0], [0.0, 2.0]], requires_grad=True)
    cls = w @ x
    loss0 = loss_of(head, cls)
    loss0.backward()
    probs = torch.softmax(head @ cls.detach(), 0)
    d_cls = head.T @ (probs - torch.tensor([1.0, 0.0]))
    by_hand = torch.outer(d_cls, x)
    agree = torch.allclose(by_hand, w.grad, atol=1e-6)
    new_w = (w - STEP * w.grad).detach()
    #endregion

    #region 2 - Error with the new W
    # what? - The CLS and the error after the step.
    # how?  - Same head, new W.
    new_cls = new_w @ x
    loss1 = loss_of(head, new_cls)
    #endregion

    #region 3 - Collect and print
    # what? - A dictionary with the numbers of the diary.
    # how?  - Tensors become lists.
    result = {"cls_before": cls.detach().tolist(), "loss_0": loss0.item(), "d_loss_d_cls": d_cls.tolist(),
              "gradient_w": by_hand.tolist(), "new_w": new_w.tolist(), "cls_after": new_cls.tolist(),
              "loss_1": loss1.item(), "gradient_by_hand_equals_autograd": agree}
    print("\nEXAMPLE 2 (session 9): one step on a body matrix, head fixed")
    print(f"  CLS before {result['cls_before']}; loss {loss0.item():.3f}")
    print(f"  gradient on the CLS {[round(v, 5) for v in result['d_loss_d_cls']]}")
    print(f"  gradient on W {[[round(v, 5) for v in r] for r in result['gradient_w']]}")
    print(f"  new W {[[round(v, 5) for v in r] for r in result['new_w']]}; new CLS {[round(v, 4) for v in result['cls_after']]}")
    print(f"  loss after the step: {loss1.item():.3f}; hand gradient == autograd: {agree}")
    return result
    #endregion


def main():
    """
    * what? - Runs both examples and writes the JSON.
    * why?  - Keeps the numbers of sessions 8 and 9 reproducible.
    * how?  - Calls the two functions and dumps their dictionaries.
    """
    #region 1 - Run and save
    # what? - exemplos_gradiente.json.
    # how?  - Keys are in English because this file is new.
    result = {"step": STEP, "example_head": example_head(), "example_body": example_body()}
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2))
    print("\nwritten:", OUTPUT.relative_to(ROOT))
    #endregion


if __name__ == "__main__":
    main()
