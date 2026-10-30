"""
* what? - Study test of session 5: the attention example with 3 tokens of 2 numbers, computed step by step.
* why?  - Every number quoted in session 5 (scores, softmax weights, new tokens) comes from this script, so it can be checked.
* how?  - q, k and v from three small matrices; scores are query times key; softmax per row; new token is the weighted sum of values.

The example leaves out two things of the real model: the division of the scores by 8 (square root of 64) and the bias numbers.
Run from the repository root:
    python3 doutorado/experimento1/etapa1/scripts/attention_example.py | tee doutorado/experimento1/etapa1/estudos/atencao_exemplo.log
Output: doutorado/experimento1/etapa1/estudos/atencao_exemplo.json
"""
import json
from pathlib import Path

import numpy as np

OUTPUT = Path(__file__).resolve().parents[1] / "estudos" / "atencao_exemplo.json"
NAMES = ["A", "B", "C"]
X = np.array([[1, 0], [0, 1], [1, 1]], float)        # tokens A, B, C
WQ = np.array([[2, 0], [0, 1]], float)
WK = np.array([[1, 0], [0, 2]], float)
WV = np.array([[1, 1], [0, 1]], float)


def softmax_rows(scores):
    """
    * what? - Turns each row of scores into percentages (between 0 and 1) that add up to 1.
    * why?  - The attention weights of a token must be positive and sum to 1.
    * how?  - exp of every score divided by the sum of the exps of its row.
    """
    #region 1 - Apply the softmax row by row
    # what? - A matrix with the same shape as scores.
    # how?  - keepdims makes the division work row by row.
    e = np.exp(scores)
    return e / e.sum(axis=1, keepdims=True)
    #endregion


def main():
    """
    * what? - Computes q, k, v, the 3 x 3 score table, the weights and the new tokens, checks the numbers of the diary and writes the JSON.
    * why?  - Replaces the quick check that was run in the terminal and not saved.
    * how?  - Matrix products and softmax in numpy; assertions compare the rounded results with the numbers printed in session 5.
    """
    #region 1 - Compute the vectors, scores, weights and new tokens
    # what? - Q, K, V (one row per token), scores = Q K', weights = softmax(scores), new = weights V.
    # how?  - The score of row i and column j is the dot product of the query of i with the key of j.
    Q, K, V = X @ WQ, X @ WK, X @ WV
    scores = Q @ K.T
    weights = softmax_rows(scores)
    new_tokens = weights @ V
    #endregion

    #region 2 - Print the tables
    # what? - The same quantities the diary quotes.
    # how?  - Rounded to 2 decimals, as in the text.
    for label, M in (("query", Q), ("key", K), ("value", V)):
        print(f"{label} of A, B, C: {M.tolist()}")
    print("\nscores (row = whose query, column = whose key):")
    for n, row in zip(NAMES, scores):
        print(f"  {n}: {row.tolist()}")
    print("\nsoftmax weights (each row sums to 1):")
    for n, row in zip(NAMES, weights):
        print(f"  {n}: {[round(float(v), 4) for v in row]}  sum = {row.sum():.4f}")
    print("\nnew tokens (weights times values):")
    for n, row in zip(NAMES, new_tokens):
        print(f"  {n}: [{row[0]:.2f} ; {row[1]:.2f}]")
    #endregion

    #region 3 - Check the numbers quoted in the diary
    # what? - Assertions against session 5.
    # how?  - Row A weights 0.47, 0.06, 0.47; new A, B, C as printed in the text.
    assert scores.tolist() == [[2, 0, 2], [0, 2, 2], [2, 2, 4]]
    assert [round(v, 2) for v in weights[0]] == [0.47, 0.06, 0.47]
    assert [[round(v, 2) for v in r] for r in new_tokens] == [[0.94, 1.47], [0.53, 1.47], [0.89, 1.79]]
    print("\nall numbers of session 5 confirmed")
    #endregion

    #region 4 - Write the result
    # what? - JSON with every table.
    # how?  - Plain lists so it can be read without numpy.
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps({"tokens": X.tolist(), "Wq": WQ.tolist(), "Wk": WK.tolist(), "Wv": WV.tolist(), "query": Q.tolist(), "key": K.tolist(), "value": V.tolist(),
                                  "scores": scores.tolist(), "weights": weights.tolist(), "new_tokens": new_tokens.tolist()}, indent=2))
    #endregion


if __name__ == "__main__":
    main()
