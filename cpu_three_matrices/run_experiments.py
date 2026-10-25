#!/usr/bin/env python3
"""CPU experiments with exact verification: three matrices, block width, RHS correlation.

Usage: python3 run_experiments.py --phase main --out results
"""

from pathlib import Path
import argparse, csv, hashlib, json, platform, sys, time, traceback, math
from fractions import Fraction as F
import numpy as np
import mpmath

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "exact"))
from verify_qr_bjac import run_case, jsonable, load_record
from certify_budgets import (
    monitor,
    validate,
    serial,
    fm,
    ss,
    root,
    solve_exact,
    energy2,
    family_constants,
)


def save(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False))
    tmp.replace(p)


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def source_hashes():
    return {
        str(p.relative_to(ROOT.parent)): sha(p)
        for p in [
            ROOT / "run_experiments.py",
            ROOT / "manifest.json",
            *sorted((ROOT.parent / "exact").glob("*.py")),
        ]
    }


def make_spec(c):
    n, s = c["n"], c["s"]
    a = np.eye(n)
    for i in range(n - 1):
        a[i, i + 1] = a[i + 1, i] = -float(
            F(c["cross"] if (i + 1) % c["blocksize"] == 0 else c["inside"])
        )
    # Generate a fixed maximal-width pool before selecting columns, so s=1,4,8 are nested.
    pool = np.random.default_rng(c["seed"]).standard_normal((n, 8))
    b = pool[:, :s].copy()
    if c["correlation_exponent"] is not None:
        b[:, 1:] = b[:, 0:1] + np.ldexp(b[:, 1:], c["correlation_exponent"])
    b = np.ldexp(b, c["scale_exponent"])
    return dict(
        name=c["id"],
        A=a,
        B=b,
        X0=np.zeros_like(b),
        n=n,
        s=s,
        blocksize=c["blocksize"],
        precision=c["precision"],
        rhs_mode="independent" if c["correlation_exponent"] is None else "near",
        k=c["p"],
        t=c["t"],
        iterations=c["steps"],
        seed=c["seed"],
    )


def summarize(c, rec, cert, v, directory):
    a, b = fm(rec["spec"]["A"]), fm(rec["spec"]["B"])
    e02 = v["initial_error_squared"]
    bn2 = ss(b)
    ap, chi = cert["constants"]["aplus"], cert["constants"]["chi"]
    start = c["steps"] // 2
    tail = cert["steps"][start:]
    dd = max(q["zeta"] for q in tail)
    ww = max(q["omega"] for q in tail)
    xx = max(q["scalars"]["x"] for q in tail)
    cc = dd + chi * xx
    level = xx + (chi * cc + root(cc * cc + (1 - chi * chi) * ww)) / (1 - chi * chi)
    ustart = tail[0]["U"]
    rows = []
    interval_checks = 0
    residual_checks = 0
    norms = [ss(b[:, j : j + 1]) for j in range(c["s"])]
    for tr, bo, va in zip(rec["steps"], cert["steps"], v["steps"]):
        k = tr["j"] + 1
        e2 = va["actual_next_energy_squared"]
        res = b - a @ fm(tr["Xnext"])
        rs2 = ss(res)
        assert rs2 <= ap * bo["Unext"] ** 2
        residual_checks += 1
        maxrel = max(float(root(ss(res[:, j : j + 1]) / norms[j])) for j in range(c["s"]))
        interval = None
        if k > start:
            ib = level + chi ** (k - start) * max(ustart - level, F(0))
            assert e2 <= ib * ib
            interval_checks += 1
            interval = float(root(ib * ib / e02))
        actual = va["actual_squared"]
        sv = tr["qr_current"]["singular_values_fp64"]
        row = dict(
            update=k,
            error_relative=float(root(e2 / e02)),
            bound_relative=float(root(bo["Unext"] ** 2 / e02)),
            baseline_relative=float(root(bo["baseline_next"] ** 2 / e02)),
            true_residual_relative=float(root(rs2 / bn2)),
            max_column_true_residual=maxrel,
            interval_bound_relative=interval,
            bound_over_error=None if not e2 else float(root(bo["Unext"] ** 2 / e2)),
            qr_input_rank=tr["qr_current"]["rank_estimate_fp64"],
            qr_input_sigma_min=sv[-1],
            qr_input_sigma_max=sv[0],
            underflow_to_zero=tr["bjac_current"]["underflow_to_zero"],
            changed_roundings=tr["bjac_current"]["rounded_values"],
            eta=float(bo["eta"]),
            eta_actual=float(root(actual["eta"])),
            gap_budget=float(bo["gamma"]),
            gap_actual=float(root(actual["gamma"])),
            ell=float(bo["ell"]),
            ell_actual=float(root(actual["ell"])),
            omega_sqrt=float(root(bo["omega"])),
            x_budget=float(bo["scalars"]["x"]),
            x_actual=float(root(actual["local_x"])),
            projected_a=float(bo["a"]),
            projected_a_actual=float(root(actual["a"])),
            projected_b=float(bo["b"]),
            projected_b_actual=float(root(actual.get("b", F(0)))),
        )
        rows.append(row)
    save(directory / "metrics.json", rows)
    with (directory / "metrics.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    eig = np.linalg.eigvalsh(rec["spec"]["A"])
    hits = [x["update"] for x in rows if x["max_column_true_residual"] <= 1e-12]
    summary = dict(
        id=c["id"],
        config=c,
        status="certified",
        steps=len(rows),
        exact_squared_comparisons=v["exact_squared_comparisons"],
        primitive_comparisons=sum(x["bjac"]["primitive_checks"] for x in cert["steps"]),
        extra_residual_checks=residual_checks,
        extra_interval_checks=interval_checks,
        chi=float(chi),
        amin=float(cert["constants"]["amin"]),
        aplus=float(ap),
        condition_number_fp64_diagnostic=float(eig[-1] / eig[0]),
        rhs_condition_number_fp64_diagnostic=float(np.linalg.cond(rec["spec"]["B"])),
        final_error_relative=rows[-1]["error_relative"],
        final_bound_relative=rows[-1]["bound_relative"],
        final_bound_over_error=rows[-1]["bound_over_error"],
        late10_error_median=float(np.median([q["error_relative"] for q in rows[-10:]])),
        first_true_residual_1e_12=hits[0] if hits else None,
        qr_input_rank_deficient_steps=sum(x["qr_input_rank"] < c["s"] for x in rows),
        underflow_to_zero=sum(x["underflow_to_zero"] for x in rows),
        interval_start=start + 1,
        interval_end=c["steps"],
        interval_level_relative=float(root(level * level / e02)),
    )
    return summary


def main():
    pa = argparse.ArgumentParser(description=__doc__)
    pa.add_argument("--phase", choices=["smoke", "pilot", "main"], default="pilot")
    pa.add_argument("--mode", choices=["all", "trace", "certify"], default="all")
    pa.add_argument("--group", choices=["A", "B", "C"])
    pa.add_argument("--case")
    pa.add_argument("--out", type=Path, default=ROOT / "results")
    pa.add_argument("--resume", action="store_true")
    pa.add_argument("--list", action="store_true")
    pa.add_argument("--limit", type=int)
    args = pa.parse_args()
    manifest = json.loads((ROOT / "manifest.json").read_text())
    cases = manifest["phases"][args.phase]
    if args.group:
        cases = [c for c in cases if args.group in c["groups"]]
    if args.case:
        cases = [c for c in cases if c["id"] == args.case]
    if args.limit is not None:
        cases = cases[: args.limit]
    if not cases:
        raise SystemExit("No matching cases")
    if args.list:
        print(json.dumps(cases, indent=2))
        return
    out = args.out / args.phase
    out.mkdir(parents=True, exist_ok=True)
    fingerprint = source_hashes()
    env = dict(
        python=sys.version,
        numpy=np.__version__,
        platform=platform.platform(),
        mpmath=mpmath.__version__,
        source_sha256=fingerprint,
        arithmetic=manifest["arithmetic"],
    )
    if (out / "environment.json").exists():
        old = json.loads((out / "environment.json").read_text())
        if old["source_sha256"] != fingerprint or any(
            old.get(k) != env.get(k) for k in ["numpy", "mpmath", "python", "platform"]
        ):
            raise SystemExit("Source or runtime changed; use a new --out directory.")
    else:
        save(out / "environment.json", env)
    for idx, c in enumerate(cases):
        d = out / c["id"]
        d.mkdir(exist_ok=True)
        trace = d / "trace.json"
        result = d / "result.json"
        if args.resume and result.exists():
            old = json.loads(result.read_text())
            if old.get("status") == "certified":
                print("skip certified", c["id"], flush=True)
                continue
            if args.mode == "trace" and old.get("status") == "trace_saved" and trace.exists():
                continue
            if old.get("status") not in ("trace_saved", "certified"):
                print("skip recorded failure", c["id"], "(omit --resume to retry)", flush=True)
                continue
        print(
            f'[{idx+1}/{len(cases)}] {c["id"]} {c["groups"]} {c["precision"]} s={c["s"]}',
            flush=True,
        )
        stage = "setup"
        t0 = time.perf_counter()
        save(d / "config.json", c)
        try:
            spec = make_spec(c)
            family_constants(fm(spec["A"]), c["blocksize"], c["p"], c["t"])
            if args.mode == "certify" or (args.resume and trace.exists()):
                stage = "load_trace"
                rec = load_record(trace)
            else:
                stage = "solver"
                rec = run_case(spec)
                save(trace, jsonable(rec))
            if args.mode == "trace":
                save(
                    result,
                    dict(
                        id=c["id"],
                        status="trace_saved",
                        config=c,
                        seconds=time.perf_counter() - t0,
                        trace_sha256=sha(trace),
                    ),
                )
                continue
            stage = "monitor"
            cert = monitor(rec)
            save(d / "certificate.json", serial(cert))
            stage = "validator"
            v = validate(rec, cert)
            save(d / "validation.json", serial(v))
            stage = "summary"
            s = summarize(c, rec, cert, v, d)
            s["seconds"] = time.perf_counter() - t0
            s["trace_sha256"] = sha(trace)
            save(result, s)
            print(
                f'  certified: {s["exact_squared_comparisons"]} squared checks; {s["seconds"]:.1f}s',
                flush=True,
            )
        except Exception as exc:
            save(
                result,
                dict(
                    id=c["id"],
                    config=c,
                    status="failed",
                    stage=stage,
                    error=repr(exc),
                    seconds=time.perf_counter() - t0,
                ),
            )
            (d / "failure.txt").write_text(traceback.format_exc())
            print("  FAILED", stage, repr(exc), flush=True)
    allrows = [json.loads(p.read_text()) for p in sorted(out.glob("*/result.json"))]
    save(out / "summary.json", allrows)
    print(
        "Results:",
        out,
        "Certified:",
        sum(x["status"] == "certified" for x in allrows),
        "Recorded failures:",
        sum(x["status"] == "failed" for x in allrows),
        flush=True,
    )
    if any(x["status"] == "failed" for x in allrows):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
