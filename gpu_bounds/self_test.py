"""Small arithmetic and model checks. Requires CUDA unless --cpu-check is given."""

import argparse, json
from fractions import Fraction as F
import numpy as np
from numerics import Backend, Matrix, BJAC
from problems_bound import problem
from solver import solve


def exact_reference(a, bs, t, p):
    n = len(a)
    ar = np.array([[F(float(v)) for v in row] for row in a], dtype=object)
    I = np.eye(n, dtype=object)
    B = np.full((n, n), F(0), dtype=object)
    for i in range(0, n, bs):
        D = np.diag([1 / ar[j, j] for j in range(i, i + bs)])
        J = np.eye(bs, dtype=object) - D @ ar[i : i + bs, i : i + bs]
        power = np.eye(bs, dtype=object)
        block = np.full((bs, bs), F(0), dtype=object)
        for _ in range(t):
            block += power @ D
            power = power @ J
        B[i : i + bs, i : i + bs] = block
    H = I - B @ ar
    power = I.copy()
    P = np.full((n, n), F(0), dtype=object)
    for _ in range(p):
        P += power @ B
        power = power @ H
    return ar, P


def tests(be):
    checks = []
    cpu = Backend("cpu")
    rng = np.random.default_rng(9001)
    # Test native operation rounding and envelopes on exactly format-representable operands.
    for fmt in ["fp64", "fp32", "fp16", "bf16"]:
        av = cpu.q(rng.uniform(-2, 2, (8, 2)), fmt)
        bv = cpu.q(rng.uniform(-2, 2, (8, 2)), fmt)
        for op in ["add", "sub", "mul"]:
            y, err = be.op(be.array(av), be.array(bv), op, fmt)
            y, err = be.host(y), be.host(err)
            for a, b, c, d in zip(av.flat, bv.flat, y.flat, err.flat):
                a, b = F(float(a)), F(float(b))
                exact = {"add": lambda: a + b, "sub": lambda: a - b, "mul": lambda: a * b}[op]()
                assert abs(F(float(c)) - exact) <= F(float(d)), (fmt, op, "primitive envelope")
    checks.append("192 primitive envelopes vs exact rational arithmetic")
    for fmt, power in [("fp16", -24), ("fp32", -149), ("bf16", -133)]:
        a = be.array(np.array([[2.0**power, 2.0**power]]))
        b = be.array(np.array([[1.0, 2.0]]))
        y, _ = be.op(a, b, "mul", fmt)
        assert np.array_equal(be.host(y), np.array([[2.0**power, 2.0 ** (power + 1)]])), (
            fmt,
            "subnormal preservation",
        )
    checks.append("FP16/FP32/BF16 subnormal preservation")
    spec = dict(
        g=2,
        rhs=2,
        family="Const",
        seed=301,
        blocksize=4,
        t=4,
        p=2,
        maxit=8,
        minit=9,
        patience=9,
        sample_every=2,
    )
    a, ix, b, xr, meta = problem(spec)
    dense = np.zeros((8, 8))
    for i in range(8):
        for j in range(7):
            dense[i, ix[i, j]] += a[i, j]
    for i in range(8):
        for col in range(2):
            assert sum(F(float(dense[i, j])) * F(float(xr[j, col])) for j in range(8)) == F(
                float(b[i, col])
            )
    ar, P = exact_reference(dense, 4, 4, 2)
    m = Matrix(a, ix, be, 4, 4, 2)
    rr = rng.standard_normal((8, 2))
    r = np.array([[F(float(v)) for v in row] for row in rr], dtype=object)
    for fmt in ["fp64", "fp32", "fp16", "bf16"]:
        z, info = BJAC(m, fmt)(be.array(rr))
        zr = np.array([[F(float(v)) for v in row] for row in be.host(z)], dtype=object)
        d = zr - P @ r
        actual = sum((d * (ar @ d)).flat, F(0))
        assert actual <= F(info["eta"]) ** 2, (fmt, "call bound")
        ans = solve(m, be.array(b), be.array(xr), BJAC(m, fmt), spec, meta, BJAC(m, "fp64"))
        assert ans["updates"] > 0 and ans["numerical_bound_violations"] == 0, (fmt, ans["status"])
        assert all(h.get("strict_certificate") is not True for h in ans["history"])
    checks.append(
        "exact manufactured solution; four call bounds vs rational polynomial; four outer trajectories"
    )
    return dict(passed=True, device=be.device, checks=checks, strict_certificate=False)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--cpu-check", action="store_true")
    args = p.parse_args()
    if args.cpu_check:
        be = Backend("cpu")
    else:
        from native_backend import NativeBackend

        be = NativeBackend()
    print(json.dumps(tests(be), indent=2))
