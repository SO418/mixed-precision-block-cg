#!/usr/bin/env python3
from pathlib import Path
import argparse, json, hashlib, time, sys, platform, traceback, csv
import numpy as np
from experiments_bound import grid
from problems_bound import problem
from numerics import Matrix, BJAC
from solver import solve

ROOT = Path(__file__).resolve().parent


def clean(x):
    if isinstance(x, dict):
        return {k: clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, np.ndarray):
        return clean(x.tolist())
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, (float, np.floating)):
        return float(x) if np.isfinite(x) else None
    if isinstance(x, np.integer):
        return int(x)
    return x


def save(p, x):
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(clean(x), indent=2, allow_nan=False))
    tmp.replace(p)


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--phase", choices=["smoke", "pilot", "main", "large"], default="pilot")
    pa.add_argument("--out", type=Path, default=ROOT / "results")
    pa.add_argument("--resume", action="store_true")
    pa.add_argument("--group", choices=["A", "B", "C"])
    pa.add_argument("--case")
    pa.add_argument("--list", action="store_true")
    args = pa.parse_args()
    jobs = grid(args.phase)
    if args.group:
        jobs = [s for s in jobs if args.group in s["groups"]]
    if args.case:
        jobs = [s for s in jobs if s["id"] == args.case]
    if not jobs:
        raise SystemExit("No selected cases")
    if args.list:
        print(json.dumps(jobs, indent=2))
        return
    from native_backend import NativeBackend
    from self_test import tests

    be = NativeBackend()
    cp = be.xp
    import cupyx

    props = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
    env = dict(
        python=sys.version,
        numpy=np.__version__,
        cupy=cp.__version__,
        runtime=cp.cuda.runtime.runtimeGetVersion(),
        driver=cp.cuda.runtime.driverGetVersion(),
        platform=platform.platform(),
        gpu=str(props["name"]),
        compute_capability=[props["major"], props["minor"]],
        source_sha256={
            p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(ROOT.glob("*.py"))
        },
        native_target_operations=True,
        fp64_transport=True,
        strict_certificate=False,
        timing_is_performance_measurement=False,
    )
    out = args.out / args.phase
    out.mkdir(parents=True, exist_ok=True)
    if (out / "environment.json").exists():
        if json.loads((out / "environment.json").read_text()) != env:
            raise SystemExit("Environment/source changed; use a new --out.")
        if not args.resume:
            raise SystemExit("Output exists; use --resume or a new --out.")
    else:
        save(out / "environment.json", env)
    print("GPU arithmetic and small-model checks...", flush=True)
    with cupyx.errstate(linalg="raise"):
        check = tests(be)
    save(out / "gpu_preflight.json", check)
    save(out / "full_plan.json", grid(args.phase))
    for i, spec in enumerate(jobs):
        target = out / "runs" / (spec["id"] + ".json")
        if args.resume and target.exists():
            print("keep recorded result", spec["id"], flush=True)
            continue
        print(
            f'[{i+1}/{len(jobs)}] {spec["id"]} {spec["family"]} n={spec["g"]**3} {spec["global_fmt"]}',
            flush=True,
        )
        start = time.perf_counter()
        try:
            a, ix, b, xr, meta = problem(spec)
            m = Matrix(a, ix, be, spec["blocksize"], spec["t"], spec["p"])
            pre = BJAC(m, spec["global_fmt"])
            ref = BJAC(m, "fp64")
            with cupyx.errstate(linalg="raise"):
                ans = solve(m, be.array(b), be.array(xr), pre, spec, meta, ref)
            result = dict(
                execution_status="completed",
                spec=spec,
                constants=m.constants(),
                reference=meta,
                result=ans,
                diagnostic_seconds=time.perf_counter() - start,
                strict_certificate=False,
            )
            print(
                " ",
                ans["status"],
                "updates",
                ans["updates"],
                "bound comparison flags",
                ans["numerical_bound_violations"],
                flush=True,
            )
        except Exception as exc:
            result = dict(
                execution_status="failed",
                spec=spec,
                error=repr(exc),
                traceback=traceback.format_exc(),
                strict_certificate=False,
            )
        save(target, result)
        cp.get_default_memory_pool().free_all_blocks()
    from summarize import summarize

    summarize(out)
    print("Results:", out, flush=True)


if __name__ == "__main__":
    main()
