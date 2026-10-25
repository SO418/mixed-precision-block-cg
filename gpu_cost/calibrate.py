"""Accuracy of isolated BJAC calls relative to an FP64 BJAC call at several input scales."""

import argparse, json, time
from pathlib import Path
import numpy as np
from model import problem, block_scores, select


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--out", required=True)
    pa.add_argument("--device", type=int, default=0)
    pa.add_argument("--g", type=int, default=16)
    pa.add_argument("--block-size", type=int, default=32)
    pa.add_argument("--t", type=int, default=4)
    pa.add_argument("--p", type=int, default=2)
    args = pa.parse_args()
    import cupy as cp
    from gpu import Kernels, BJAC
    from self_test import gpu_tests

    cp.cuda.Device(args.device).use()
    info = gpu_tests()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    (out / "environment.json").write_text(json.dumps(dict(info, args=vars(args)), indent=2))
    k = Kernels()
    records = []
    for kind in ["Const", "Dis", "Ani"]:
        a, ix = problem(args.g, kind, 1000, args.block_size)
        ag, ig = cp.asarray(a), cp.asarray(ix)
        scores, theta = block_scores(a, ix, args.block_size, args.t)
        n = len(a)
        base = np.random.default_rng(71).normal(size=(n, 4))
        reference = BJAC(
            k, ag, ig, args.block_size, args.t, args.p, np.zeros(n // args.block_size, bool), True
        )
        for policy in ["fp32", "fp16", "weighted", "worst", "random_matched"]:
            mask, stats = select(scores, policy, 0.06)
            pre = BJAC(k, ag, ig, args.block_size, args.t, args.p, mask)
            for scale in [1.0, 1e-4, 1e-8, 1e-12]:
                r = cp.asarray(base * scale)
                ref = reference(r)
                got = pre(r)
                rel = cp.asnumpy(
                    cp.linalg.norm(got - ref, axis=0) / cp.linalg.norm(ref, axis=0)
                ).tolist()
                records.append(
                    dict(
                        kind=kind,
                        policy=policy,
                        scale=scale,
                        selection=stats,
                        relative_difference_from_fp64_per_column=rel,
                        input_fraction_nonzero_rounded_to_half_zero=float(
                            np.mean((base * scale != 0) & ((base * scale).astype(np.float16) == 0))
                        ),
                        note="relative 2-norm difference from the FP64 BJAC call",
                    )
                )
    (out / "call_diagnostics.json").write_text(
        json.dumps(records, indent=2, allow_nan=False) + "\n"
    )
    print("Saved BJAC call diagnostics:", out)


if __name__ == "__main__":
    main()
