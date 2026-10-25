"""Preconditioner and outer error bounds on four controlled n = 48 cases (12 updates each).

Usage: python3 outer_budget.py --out results
"""

import argparse, json, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "exact"))
from verify_qr_bjac import run_case, jsonable
from certify_budgets import monitor, validate, serial


def main(root):
    out = root / "outer_cases"
    out.mkdir(parents=True, exist_ok=True)
    summary = []
    for s, fmt, near in [
        (1, "fp16", False),
        (4, "fp16", False),
        (4, "fp16", True),
        (4, "fp32", False),
    ]:
        name = f"n48_s{s}_{fmt}" + ("_near" if near else "")
        n = 48
        bs = 4
        a = np.eye(n)
        for j in range(n - 1):
            a[j, j + 1] = a[j + 1, j] = -(1 / 32 if (j + 1) % bs == 0 else 1 / 4)
        b = np.random.default_rng(20260913).standard_normal((n, s))
        if near:
            b[:, 1:] = b[:, 0:1] + 1e-10 * b[:, 1:]
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
            iterations=12,
            seed=20260913,
        )
        rec = run_case(spec)
        (out / (name + "_trace.json")).write_text(json.dumps(jsonable(rec)))
        cert = monitor(rec)
        v = validate(rec, cert)
        (out / (name + "_certificate.json")).write_text(json.dumps(serial(cert)))
        (out / (name + "_validation.json")).write_text(json.dumps(serial(v)))
        summary.append(
            {"name": name, "validation": serial(v), "constants": serial(cert["constants"])}
        )
        print(name, v["exact_squared_comparisons"], flush=True)
    (root / "outer_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    pa = argparse.ArgumentParser()
    pa.add_argument("--out", type=Path, default=Path("results"))
    main(pa.parse_args().out)
