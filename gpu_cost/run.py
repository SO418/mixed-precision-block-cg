"""GPU accuracy and cost study: FP64/FP32/FP16, call scaling, local scaling and switching.

Usage: python3 run.py --preset core --out results/core
Runs with --diagnostics or --profile are not used for timings.
"""

import argparse, json, time, hashlib, itertools, math
from pathlib import Path
import numpy as np
from model import problem
from precision_modes import PrecisionBJAC


def configurations(preset, seeds):
    if preset == "smoke":
        seeds = [301]
    methods = [
        ("fp64", None),
        ("fp32", None),
        ("fp16", None),
        ("call_scaled", None),
        ("local_scaled", None),
        ("switch", 1e-4),
    ]
    tp = [(4, 2)]
    scales = [0]
    kinds = ["Const", "Dis", "Ani"]
    if preset == "smoke":
        kinds = ["Const"]
    if preset == "thresholds":
        methods = [("fp64", None), ("fp32", None)] + [
            ("switch", v) for v in [1e-2, 1e-4, 1e-6, 1e-8]
        ]
    if preset == "work":
        tp = list(itertools.product([2, 4, 8], [1, 2, 4]))
        methods = [("fp64", None), ("fp32", None), ("local_scaled", None), ("switch", 1e-4)]
    if preset == "scales":
        scales = [0, -20, -40]
    return [
        dict(kind=k, seed=seed, t=t, p=p, method=m, threshold=th, rhs_power2=e)
        for k, seed, (t, p), (m, th), e in itertools.product(kinds, seeds, tp, methods, scales)
    ]


def safe(v):
    if isinstance(v, dict):
        return {k: safe(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [safe(x) for x in v]
    if isinstance(v, float) and not math.isfinite(v):
        return None
    return v


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument(
        "--preset", choices=["smoke", "core", "thresholds", "work", "scales"], default="core"
    )
    pa.add_argument("--out", required=True)
    pa.add_argument("--device", type=int, default=0)
    pa.add_argument("--seeds", type=int, nargs="+", default=[301, 302, 303])
    pa.add_argument("--repeat", type=int, default=3)
    pa.add_argument("--g", type=int)
    pa.add_argument("--rhs", type=int, default=8)
    pa.add_argument("--block-size", type=int, default=32)
    pa.add_argument("--tol", type=float, default=1e-12)
    pa.add_argument("--maxit", type=int, default=400)
    pa.add_argument("--check-every", type=int, default=5)
    pa.add_argument("--diagnostics", action="store_true")
    pa.add_argument("--profile", action="store_true")
    args = pa.parse_args()
    if (
        args.repeat < 1
        or args.rhs < 1
        or args.block_size < 1
        or args.maxit < 1
        or args.check_every < 1
        or not 0 < args.tol < 1
    ):
        raise ValueError("invalid arguments")
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("duplicate seeds")
    g = args.g or (8 if args.preset == "smoke" else 48)
    if g**3 % args.block_size or args.rhs * args.block_size > 1024 or args.rhs > g**3:
        raise ValueError("invalid grid/block/RHS dimensions")
    import cupy as cp
    from gpu import Kernels, solve, source
    from checks import cpu_checks, gpu_checks

    cp.cuda.Device(args.device).use()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    env = cpu_checks()
    env.update(gpu_checks())
    env.update(
        args=vars(args),
        g=g,
        n=g**3,
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        performance_eligible=not (args.diagnostics or args.profile),
        note="Scaling and switching are heuristics without a convergence guarantee.",
        source_sha256={
            f.name: hashlib.sha256(f.read_bytes()).hexdigest()
            for f in Path(__file__).parent.glob("*.py")
        },
    )
    (out / "environment.json").write_text(json.dumps(env, indent=2) + "\n")
    (out / "kernels.cu").write_text(source())
    cfg = configurations(args.preset, args.seeds)
    order = np.random.default_rng(20260913).permutation(len(cfg))
    cfg = [cfg[i] for i in order]
    (out / "schedule.json").write_text(json.dumps(cfg, indent=2) + "\n")
    k = Kernels()
    cache = {}
    with (out / "results.jsonl").open("a", buffering=1) as sink:
        for index, c in enumerate(cfg):
            key = c["kind"]
            if key not in cache:
                cache.clear()
                a, ix = problem(g, key, 1000, args.block_size)
                cache[key] = (cp.asarray(a), cp.asarray(ix))
            ag, ig = cache[key]
            b = cp.asarray(
                np.ldexp(
                    np.random.default_rng(c["seed"]).normal(size=(g**3, args.rhs)), c["rhs_power2"]
                )
            )
            bn = cp.linalg.norm(b, axis=0)

            def make(diag=False):
                return PrecisionBJAC(
                    k,
                    ag,
                    ig,
                    args.block_size,
                    c["t"],
                    c["p"],
                    c["method"],
                    bn,
                    cp,
                    threshold=c["threshold"] or 1e-4,
                    diagnostics=diag,
                )

            pre = make()
            warm = solve(k, ag, ig, b, pre, tol=args.tol, maxit=3, check_every=3)
            del pre
            print(f"[{index+1}/{len(cfg)}] {c}", flush=True)
            for rep in range(args.repeat):
                cp.cuda.get_current_stream().synchronize()
                start = time.perf_counter()
                pre = make(args.diagnostics)
                cp.cuda.get_current_stream().synchronize()
                setup = time.perf_counter() - start
                r = solve(
                    k,
                    ag,
                    ig,
                    b,
                    pre,
                    tol=args.tol,
                    maxit=args.maxit,
                    check_every=args.check_every,
                    profile=args.profile,
                )
                r.update(
                    config=c,
                    repeat=rep,
                    setup_wall_seconds=setup,
                    setup_plus_solve_seconds=setup + r["solve_wall_seconds"],
                    preconditioner=pre.statistics(),
                    warmup_status=warm["status"],
                    tol=args.tol,
                    n=g**3,
                    s=args.rhs,
                    block_size=args.block_size,
                    profile=args.profile,
                    diagnostics=args.diagnostics,
                    performance_eligible=env["performance_eligible"],
                )
                if args.diagnostics:
                    name = f"conversion_trace_{index:03d}_{rep}.json"
                    (out / name).write_text(json.dumps(safe(pre.trace), allow_nan=False) + "\n")
                    r["conversion_trace"] = name
                sink.write(json.dumps(safe(r), allow_nan=False) + "\n")
                print(
                    rep,
                    r["status"],
                    r["iterations"],
                    f"{r['solve_wall_seconds']:.4f}s",
                    pre.statistics(),
                    flush=True,
                )
                del pre
    from summarize import summarize

    summarize(out)


if __name__ == "__main__":
    main()
