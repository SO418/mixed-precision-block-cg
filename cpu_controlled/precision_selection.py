"""Blockwise FP16/FP32 assignments for a nonunit-diagonal example (all 16 candidates).

Four independent 2x2 blocks A_i = d_i [[1, -c_i], [-c_i, 1]], one complete correction with
two local steps. For each assignment the structured bound and the comparator are checked
against the exact call error in rational arithmetic, and the smallest number of FP32 blocks
meeting each error target is recorded.

Usage: python3 precision_selection.py --out results
"""

import argparse, itertools, json, sys
from fractions import Fraction as F
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "exact"))
from certify_budgets import LowMonitor, fm, root, ss, energy2, up

D = [1, 4, 16, 64]
cs = [F(1, 4), F(1, 8), F(1, 4), F(1, 8)]
r = np.array(
    [
        1.234567,
        -0.765432,
        0.154320875,
        -0.095679,
        0.019290109375,
        -0.011959875,
        0.002411263671875,
        -0.001494984375,
    ]
)[:, None]
beta = F(17, 16)


def main(out):
    blocks = []
    weights = []
    local = []
    for i, (d, c) in enumerate(zip(D, cs)):
        a = np.array([[d, -float(c) * d], [-float(c) * d, d]], float)
        ar = fm(a)
        ri = r[2 * i : 2 * i + 2]
        rr = fm(ri)
        J = np.eye(2, dtype=object) - ar / F(d)
        B = (np.eye(2, dtype=object) + J) / F(d)
        ref = B @ rr
        weight = (rr.T @ B @ rr)[0, 0]
        weights.append(weight)
        results = {}
        for fmt in ["fp16", "fp32"]:
            mon = LowMonitor(fmt)
            u = np.zeros((2, 1))
            L = F(0)
            for q in range(2):
                w, ew = mon.mm(a, u)
                v, ev = mon.op(ri, w, "-")
                corr, ec = mon.op(np.full((2, 1), 1 / d), v, "*")
                un, ea = mon.op(u, corr, "+")
                dq = ew / F(d) + ev / F(d) + ec + ea
                L = up(L + c ** (1 - q) * root(F(d) * ss(dq) / ((1 - c * c) / (1 + c))))
                u = un
            err2 = energy2(ar, fm(u) - ref)
            results[fmt] = {"L": L, "z": fm(u), "checks": mon.checks, "error2": err2}
        local.append(results)
        blocks.append((ar, ref))
    rows = []
    for bits in itertools.product([0, 1], repeat=4):
        chosen = [local[i]["fp32" if b else "fp16"] for i, b in enumerate(bits)]
        ls = [x["L"] for x in chosen]
        bound = root(beta * sum(l * l for l in ls))
        # Upper square root of the ratio; avoids dividing by a rounded-up square root.
        coarse = root(beta * max(l * l / w for l, w in zip(ls, weights)) * sum(weights))
        error2 = sum(x["error2"] for x in chosen)
        assert error2 <= bound * bound and bound <= coarse
        rows.append(
            {
                "mask": "".join(map(str, bits)),
                "fp32_blocks": sum(bits),
                "bound": float(bound),
                "coarse": float(coarse),
                "actual_error": float(root(error2)),
                "bound_rational": str(bound),
                "coarse_rational": str(coarse),
                "error_squared_rational": str(error2),
            }
        )
    targets = [F(1, 10**j) for j in [2, 3, 4, 5, 6]]
    decisions = []
    for tol in targets:

        def choose(field):
            good = [x for x in rows if F(x[field + "_rational"]) <= tol]
            return (
                min(good, key=lambda x: (x["fp32_blocks"], F(x[field + "_rational"]), x["mask"]))
                if good
                else None
            )

        decisions.append(
            {"tolerance": str(tol), "structured": choose("bound"), "coarse": choose("coarse")}
        )
    result = {
        "n": 8,
        "blocksize": 2,
        "t": 2,
        "p": 1,
        "D": D,
        "couplings": [str(c) for c in cs],
        "rhs": r.tolist(),
        "beta": str(beta),
        "rho": str(F(1, 16)),
        "primitive_checks": sum(x["checks"] for l in local for x in l.values()),
        "policies": rows,
        "decisions": decisions,
    }
    (out / "precision_selection.json").write_text(json.dumps(result, indent=2))
    for d in decisions:
        print(
            d["tolerance"], [(k, d[k]["mask"] if d[k] else None) for k in ["structured", "coarse"]]
        )


if __name__ == "__main__":
    pa = argparse.ArgumentParser()
    pa.add_argument("--out", type=Path, default=Path("results"))
    args = pa.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    main(args.out)
