#!/usr/bin/env python3
"""Order-16 example with four inner formats (finite-interval bound)."""

import argparse, json, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "exact"))
from verify_qr_bjac import run_case, jsonable
from certify_budgets import monitor, validate, serial


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--out", type=Path, default=Path("results/exact"))
    pa.add_argument("--profile", choices=["smoke", "full"], default="full")
    pa.add_argument("--resume", action="store_true")
    args = pa.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    cases = [
        (16, 4, fmt, near, 0, 4 if args.profile == "smoke" else 50)
        for near in [False, True]
        for fmt in ["fp64", "fp32", "fp16", "bf16"]
    ]
    import platform, hashlib

    env = dict(
        python=platform.python_version(),
        numpy=np.__version__,
        platform=platform.platform(),
        arithmetic="CPU, explicit rounding to the target format; BF16 via FP32",
        source_sha256={
            f.name: hashlib.sha256(f.read_bytes()).hexdigest()
            for f in [*ROOT.glob("*.py"), *(ROOT.parent / "exact").glob("*.py")]
        },
    )
    (args.out / "environment.json").write_text(json.dumps(env, indent=2))
    rows = []
    for n, s, fmt, near, scale, steps in cases:
        name = f"n{n}_s{s}_{fmt}_near{int(near)}_scale{scale}_steps{steps}"
        target = args.out / (name + "_result.json")
        if args.resume and target.exists():
            rows.append(json.loads(target.read_text()))
            continue
        print(name, flush=True)
        start = time.perf_counter()
        a = np.eye(n)
        bs = 4
        for i in range(n - 1):
            a[i, i + 1] = a[i + 1, i] = -(1 / 32 if (i + 1) % bs == 0 else 1 / 4)
        b = np.random.default_rng(301).standard_normal((n, s))
        if near and s > 1:
            b[:, 1:] = b[:, 0:1] + 2.0**-30 * b[:, 1:]
        b = np.ldexp(b, scale)
        spec = dict(
            name=name,
            A=a,
            B=b,
            X0=np.zeros_like(b),
            n=n,
            s=s,
            blocksize=bs,
            precision=fmt,
            rhs_mode="near" if near else "random",
            k=2,
            t=3,
            iterations=steps,
            seed=301,
        )
        rec = run_case(spec)
        (args.out / (name + "_trace.json")).write_text(json.dumps(jsonable(rec)))
        cert = monitor(rec)
        v = validate(rec, cert)
        (args.out / (name + "_certificate.json")).write_text(json.dumps(serial(cert)))
        row = dict(
            name=name,
            strict_certificate=True,
            validation=serial(v),
            constants=serial(cert["constants"]),
            seconds=time.perf_counter() - start,
        )
        target.write_text(json.dumps(row))
        rows.append(row)
        print("  exact comparisons:", v["exact_squared_comparisons"], flush=True)
    (args.out / "summary.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
