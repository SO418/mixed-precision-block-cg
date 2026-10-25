"""CPU and GPU self-tests for the kernels."""

import argparse, json, platform
from pathlib import Path
import numpy as np
from model import problem, dense, block_scores, select, cpu_local


def cpu_tests():
    for kind in ["Const", "Dis", "Ani"]:
        a, ix = problem(4, kind, 100.0, 8)
        aa = dense(a, ix)
        assert np.allclose(aa, aa.T, rtol=0, atol=1e-15)
        assert np.linalg.eigvalsh(aa)[0] > 0
        assert np.all(np.diag(aa) == 1)
    scores = {16: np.array([0.01, 0.03, 0.1, 0.3]), 32: np.full(4, 1e-5)}
    mask, stats = select(scores, "weighted", 0.06)
    assert stats["proxy_target_met"] and not mask[-1] and mask[0]
    random, _ = select(scores, "random_matched", 0.06)
    assert random.sum() == mask.sum()
    assert select(scores, "worst", 0.06)[0].sum() == 0
    a, ix = problem(4, "Const", 1, 8)
    r = np.random.default_rng(9).normal(size=(64, 2))
    z = cpu_local(a, ix, r, 8, 3, np.zeros(8, bool), np.float64)
    ref = np.empty_like(r)
    aa = dense(a, ix)
    for beg in range(0, 64, 8):
        block = aa[beg : beg + 8, beg : beg + 8]
        j = np.eye(8) - block
        ref[beg : beg + 8] = (np.eye(8) + j + j @ j) @ r[beg : beg + 8]
    assert np.allclose(z, ref, rtol=1e-13, atol=1e-13)
    return {
        "cpu_checks": "passed",
        "matrix_families": 3,
        "policy_checks": "passed",
        "reference_polynomial": "passed",
    }


def gpu_tests():
    import cupy as cp
    from gpu import Kernels, BJAC, solve, source

    k = Kernels()
    a, ix = problem(4, "Const", 1, 8)
    ag, ig = cp.asarray(a), cp.asarray(ix)
    rng = np.random.default_rng(2)
    r = rng.normal(size=(64, 2))
    for dtype in [np.float32, np.float64]:
        got = cp.asnumpy(k.sp(ag.astype(dtype), ig, cp.asarray(r, dtype=dtype)))
        np.testing.assert_allclose(
            got,
            dense(a, ix) @ r,
            rtol=2e-6 if dtype == np.float32 else 1e-13,
            atol=2e-6 if dtype == np.float32 else 1e-13,
        )
    for policy in ["fp16", "fp32", "fp64", "mixed"]:
        mask = np.ones(8, bool) if policy == "fp16" else np.zeros(8, bool)
        if policy == "mixed":
            mask[::2] = True
        dt = np.float64 if policy == "fp64" else np.float32
        pre = BJAC(k, ag, ig, 8, 3, 1, mask, policy == "fp64")
        got = cp.asnumpy(pre(cp.asarray(r)))
        expected = cpu_local(a, ix, r.astype(dt), 8, 3, mask, dt)
        if policy in ("fp16", "mixed"):
            np.testing.assert_array_equal(got, expected)
        else:
            np.testing.assert_allclose(
                got,
                expected,
                rtol=1e-13 if dt == np.float64 else 1e-6,
                atol=1e-13 if dt == np.float64 else 1e-6,
            )
    # Exercise p>1, complete correction order, and mixed grouping.
    mask = np.array([True, False] * 4)
    pre = BJAC(k, ag, ig, 8, 3, 2, mask)
    rr = r.astype(np.float32)
    y = np.zeros_like(rr)
    af = a.astype(np.float32)

    def cpu_sp(x):
        out = np.zeros_like(x)
        for j in range(7):
            out = np.float32(out + np.float32(af[:, j, None] * x[ix[:, j]]))
        return out

    for _ in range(2):
        residual = np.float32(rr - cpu_sp(y))
        y = np.float32(y + cpu_local(a, ix, residual, 8, 3, mask, np.float32))
    np.testing.assert_array_equal(cp.asnumpy(pre(cp.asarray(r))), y.astype(float))
    pre = BJAC(k, ag, ig, 8, 3, 2, np.zeros(8, bool), True)
    b = cp.asarray(r)
    res = solve(k, ag, ig, b, pre, tol=1e-9, maxit=100, check_every=1)
    assert res["status"] == "converged", res
    # Repeated and nearly correlated RHS exercise the raw QR completion path.
    for epsilon in [0.0, 1e-10]:
        b = cp.asarray(np.column_stack([r[:, 0], r[:, 0] + epsilon * r[:, 1]]))
        pre = BJAC(k, ag, ig, 8, 3, 2, np.zeros(8, bool), True)
        res = solve(k, ag, ig, b, pre, tol=1e-8, maxit=150, check_every=1)
        assert res["status"] == "converged", res
    prop = cp.cuda.runtime.getDeviceProperties(cp.cuda.runtime.getDevice())
    name = prop["name"]
    name = name.decode() if isinstance(name, bytes) else name
    return {
        "gpu_checks": "passed",
        "gpu": name,
        "cupy": cp.__version__,
        "runtime": cp.cuda.runtime.runtimeGetVersion(),
        "driver": cp.cuda.runtime.driverGetVersion(),
        "python": platform.python_version(),
        "numpy": np.__version__,
    }


if __name__ == "__main__":
    pa = argparse.ArgumentParser()
    pa.add_argument("--gpu", action="store_true")
    pa.add_argument("--out", default="self_test.json")
    args = pa.parse_args()
    result = cpu_tests()
    if args.gpu:
        result.update(gpu_tests())
    else:
        result["gpu_checks"] = "not run (CPU only)"
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
