"""CPU and GPU checks for the precision variants."""

import argparse, json
from pathlib import Path
import numpy as np
from model import problem, cpu_local
from precision_modes import PrecisionBJAC, normalize_columns


class CPUKernels:
    def sp(self, a, ix, x):
        out = np.zeros_like(x)
        for j in range(7):
            out = out + (a[:, j, None] * x[ix[:, j]])
        return out

    def local(self, a, ix, r, bs, t, groups):
        bits = next(iter(groups))
        return cpu_local(
            a[bits].astype(float),
            ix,
            r,
            bs,
            t,
            np.full(len(r) // bs, bits == 16),
            np.float64 if bits == 64 else np.float32,
        )


def cpu_checks():
    x = np.array([[2.0**-990, 2.0**990, 0.0], [-(2.0**-991), 2.0**989, 0.0]])
    scaled, e = normalize_columns(x, np)
    np.testing.assert_array_equal(np.ldexp(scaled, e[None, :]), x)
    assert np.all(np.max(np.abs(scaled), axis=0) == [0.5, 0.5, 0])
    a, ix = problem(4, "Const", 1, 8)
    r = np.random.default_rng(11).normal(size=(64, 2))
    bn = np.linalg.norm(r, axis=0)
    for method in ["call_scaled", "local_scaled"]:
        pre = PrecisionBJAC(CPUKernels(), a, ix, 8, 3, 2, method, bn, np, diagnostics=True)
        large = pre(r)
        small = pre(np.ldexp(r, -40))
        np.testing.assert_allclose(np.ldexp(small, 40), large, rtol=1e-6, atol=1e-6)
        assert all(max(z["zero_loss_at_kernel"]) == 0 for z in pre.trace)
        np.testing.assert_array_equal(pre(np.zeros_like(r)), np.zeros_like(r))
    pre = PrecisionBJAC(CPUKernels(), a, ix, 8, 3, 2, "switch", bn, np, threshold=1e-4)
    pre(r)
    assert not pre.switched and pre.half_calls == 1
    small = r.copy()
    small[:, 0] *= 1e-6
    pre(small)
    assert pre.switched and pre.switch_call == 2 and pre.single_calls == 1
    pre(r)
    assert pre.single_calls == 2 and pre.half_calls == 1
    return {
        "variants_cpu": "passed",
        "scale_roundtrip": "passed",
        "scale_covariance": "passed",
        "sticky_switch": "passed",
    }


def gpu_checks():
    import cupy as cp
    from gpu import Kernels, BJAC, solve
    from self_test import gpu_tests

    info = gpu_tests()
    k = Kernels()
    a, ix = problem(4, "Const", 1, 8)
    ag, ig = cp.asarray(a), cp.asarray(ix)
    r = np.random.default_rng(11).normal(size=(64, 2))
    rg = cp.asarray(r)
    bn = cp.linalg.norm(rg, axis=0)
    for method in ["fp64", "fp32", "fp16"]:
        old = BJAC(k, ag, ig, 8, 3, 2, np.full(8, method == "fp16"), method == "fp64")
        new = PrecisionBJAC(k, ag, ig, 8, 3, 2, method, bn, cp)
        np.testing.assert_array_equal(cp.asnumpy(old(rg)), cp.asnumpy(new(rg)))
    for method in ["call_scaled", "local_scaled"]:
        cg = PrecisionBJAC(k, ag, ig, 8, 3, 2, method, bn, cp, diagnostics=True)
        cc = PrecisionBJAC(CPUKernels(), a, ix, 8, 3, 2, method, np.linalg.norm(r, axis=0), np)
        tiny = np.ldexp(r, -40)
        np.testing.assert_allclose(
            cp.asnumpy(cg(cp.asarray(tiny))), cc(tiny), rtol=1e-6, atol=1e-18
        )
        assert all(max(z["zero_loss_at_kernel"]) == 0 for z in cg.trace)
    results = {}
    for method in ["fp64", "fp32", "call_scaled", "local_scaled", "switch"]:
        pre = PrecisionBJAC(k, ag, ig, 8, 3, 2, method, bn, cp, threshold=1e-3)
        res = solve(k, ag, ig, rg, pre, tol=1e-10, maxit=150, check_every=1)
        assert res["status"] == "converged", (method, res)
        if method == "switch":
            assert pre.switched
        results[method] = res["max_true_relative_residual"]
    info.update(
        variants_gpu="passed", raw_baseline_parity="bitwise passed", small_solve_residuals=results
    )
    return info


if __name__ == "__main__":
    pa = argparse.ArgumentParser()
    pa.add_argument("--gpu", action="store_true")
    pa.add_argument("--out", default="checks.json")
    a = pa.parse_args()
    result = cpu_checks()
    if a.gpu:
        result.update(gpu_checks())
    else:
        result["variants_gpu"] = "not run"
    Path(a.out).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
