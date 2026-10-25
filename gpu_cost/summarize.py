"""Summarize repeats within a seed, then pair methods on identical problems."""

import argparse, csv, json, statistics as st
from pathlib import Path


def summarize(folder):
    folder = Path(folder)
    env = json.loads((folder / "environment.json").read_text())
    expected = env["args"]["repeat"]
    groups = {}
    for line in (folder / "results.jsonl").read_text().splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        groups.setdefault(json.dumps(r["config"], sort_keys=True), []).append(r)
    rows = []
    for rs in groups.values():
        r = rs[0]
        good = [
            v
            for v in rs
            if v["status"] == "converged"
            and v["max_true_relative_residual"] is not None
            and v["max_true_relative_residual"] <= v["tol"]
        ]
        complete = len(rs) == expected and {v["repeat"] for v in rs} == set(range(expected))
        valid = complete and len(good) == expected
        row = dict(
            r["config"],
            n=r["n"],
            s=r["s"],
            tol=r["tol"],
            runs=len(rs),
            successes=len(good),
            valid=valid,
            performance_eligible=r["performance_eligible"],
        )
        for name in ["solve_wall_seconds", "setup_plus_solve_seconds", "iterations"]:
            row[name] = st.median(x[name] for x in good) if valid else None
        for name in ["half_calls", "single_calls", "double_calls"]:
            row[name] = st.median(v["preconditioner"][name] for v in rs)
        sc = [v["preconditioner"]["switch_call"] for v in rs]
        row["switch_call"] = st.median(sc) if all(x is not None for x in sc) else None
        vals = [
            v["max_true_relative_residual"]
            for v in rs
            if v["max_true_relative_residual"] is not None
        ]
        row["terminal_residual_median"] = st.median(vals) if vals else None
        row["paired_speedup_vs_fp32"] = None
        row["paired_speedup_vs_fp64"] = None
        rows.append(row)

    def key(r):
        return tuple(r[k] for k in ["kind", "seed", "t", "p", "rhs_power2", "n", "s", "tol"])

    bases = {
        (key(r), r["method"]): r
        for r in rows
        if r["valid"] and r["performance_eligible"] and r["method"] in ("fp32", "fp64")
    }
    for r in rows:
        if r["valid"] and r["performance_eligible"]:
            for method in ["fp32", "fp64"]:
                b = bases.get((key(r), method))
                if b:
                    r["paired_speedup_vs_" + method] = (
                        b["setup_plus_solve_seconds"] / r["setup_plus_solve_seconds"]
                    )
    rows.sort(
        key=lambda r: (
            r["kind"],
            r["seed"],
            r["rhs_power2"],
            r["t"],
            r["p"],
            r["method"],
            r["threshold"] or 0,
        )
    )
    if rows:
        with (folder / "summary.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    lines = [
        "# GPU accuracy and cost",
        "",
        "One row per RHS seed. Times are reported only if every repeat reaches the tolerance.",
        "",
        "|matrix|seed|method|threshold|RHS exponent|success/runs|updates|setup+solve (s)|vs FP32|",
        "|---|---:|---|---|---:|---|---:|---:|---:|",
    ]
    fmt = lambda x: "—" if x is None else f"{x:.5g}"
    for r in rows:
        lines.append(
            f"|{r['kind']}|{r['seed']}|{r['method']}|{fmt(r['threshold'])}|{r['rhs_power2']}|{r['successes']}/{r['runs']}|{fmt(r['iterations'])}|{fmt(r['setup_plus_solve_seconds'])}|{fmt(r['paired_speedup_vs_fp32'])}|"
        )
    (folder / "summary.md").write_text("\n".join(lines) + "\n")
    print("Saved summaries:", folder)
    return rows


if __name__ == "__main__":
    pa = argparse.ArgumentParser()
    pa.add_argument("folder")
    summarize(pa.parse_args().folder)
