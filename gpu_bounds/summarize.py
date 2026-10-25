"""Summarize all runs and plot seed 301."""

from pathlib import Path
import argparse, csv, json, math
import numpy as np


def summarize(root):
    root = Path(root)
    data = [json.loads(p.read_text()) for p in sorted((root / "runs").glob("*.json"))]
    rows = []
    for d in data:
        s = d["spec"]
        a = d.get("result", {})
        h = a.get("history", [])
        last = h[-1] if h else {}
        valid = [x for x in h if x.get("bound_prefix_available") is True]
        row = dict(
            id=s["id"],
            groups=",".join(s["groups"]),
            family=s["family"],
            n=s["g"] ** 3,
            rhs=s["rhs"],
            corr_exp=s.get("corr_exp"),
            rhs_exp=s["rhs_exp"],
            seed=s["seed"],
            precision=s["global_fmt"],
            execution=d["execution_status"],
            solver_status=a.get("status"),
            updates=a.get("updates"),
            chi=d.get("constants", {}).get("chi"),
            final_error_A=last.get("forward_A_reference"),
            final_bound_A=last.get("bound_A"),
            final_bound_over_error=last.get("bound_over_forward_error"),
            finite_bound_steps=len(valid),
            first_unavailable_step=next(
                (x["iteration"] for x in h if x.get("bound_prefix_available") is False), None
            ),
            bound_comparison_flags=a.get("numerical_bound_violations"),
            call_comparison_flags=a.get("application_bound_violations"),
            first_target=a.get("first_target_iteration"),
            diagnostic_seconds=d.get("diagnostic_seconds"),
            strict_certificate=False,
        )
        rows.append(row)
    with (root / "summary.csv").open("w", newline="") as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    (root / "summary.json").write_text(json.dumps(rows, indent=2))
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "pdf.fonttype": 42})
    figdir = root / "figures"
    figdir.mkdir(exist_ok=True)
    groups = {}
    for d in data:
        s = d["spec"]
        if s["seed"] != 301 or d.get("execution_status") != "completed":
            continue
        key = (s["family"], s["g"], s["rhs"], s.get("corr_exp"), s["rhs_exp"])
        groups.setdefault(key, []).append(d)
    colors = {"fp64": "#283E60", "fp32": "#176A8C", "fp16": "#C45A2A", "bf16": "#805A9E"}
    for key, ds in groups.items():
        fig, ax = plt.subplots(1, 3, figsize=(12, 3.5), layout="constrained")
        for d in sorted(ds, key=lambda d: list(colors).index(d["spec"]["global_fmt"])):
            fmt = d["spec"]["global_fmt"]
            co = colors[fmt]
            hist = d["result"]["history"]
            e0 = hist[0]["forward_A_reference"]
            steps = np.array([x["iteration"] for x in hist])
            cv = lambda name: np.array(
                [x.get(name) if x.get(name) is not None else np.nan for x in hist]
            )
            ax[0].semilogy(steps, cv("forward_A_reference") / e0, color=co, label=fmt.upper())
            ax[0].semilogy(steps, cv("bound_A") / e0, color=co, ls="--")
            ax[0].semilogy(steps, cv("gap_baseline_A") / e0, color=co, ls=":", alpha=0.7)
            ax[1].semilogy(steps, cv("bound_over_forward_error"), color=co)
            calls = [x for x in d["result"]["calls"] if "reference_difference_A" in x]
            if calls:
                k = [x["call"] for x in calls]
                ax[2].semilogy(
                    k,
                    [
                        x["reference_difference_A"] if x["reference_difference_A"] > 0 else np.nan
                        for x in calls
                    ],
                    color=co,
                    marker="o",
                    markersize=3,
                )
                ax[2].semilogy(
                    k,
                    [x["eta"] + x["reference_eta"] for x in calls],
                    color=co,
                    ls="--",
                    marker="s",
                    markersize=3,
                )
        ax[0].set_title("Energy error and bounds")
        ax[0].set_ylabel("Value / initial energy error")
        ax[0].legend(fontsize=7)
        from matplotlib.lines import Line2D

        ax[1].legend(
            handles=[
                Line2D([], [], color="black", label="Measured error"),
                Line2D([], [], color="black", ls="--", label="Recurrence bound"),
                Line2D([], [], color="black", ls=":", label="Residual-gap bound"),
            ],
            title="Line styles in left panel",
            fontsize=6,
            title_fontsize=7,
        )
        ax[2].legend(
            handles=[
                Line2D([], [], color="black", marker="o", label="Distance to FP64 call"),
                Line2D(
                    [], [], color="black", ls="--", marker="s", label="Call budgets: low + FP64"
                ),
            ],
            fontsize=6,
        )
        ax[1].set_title("Bound / measured energy error")
        ax[1].axhline(1, color="black", lw=0.7)
        ax[1].set_ylabel("Ratio")
        ax[2].set_title("Same-input preconditioner comparison")
        ax[2].set_ylabel("Absolute energy norm")
        for a in ax:
            a.set_xlabel("Iteration / call index")
            a.grid(alpha=0.2)
        family, g, s, corr, scale = key
        fig.suptitle(
            f"{family}, n={g**3}, s={s}, correlation exponent={corr}, scale=2^{scale}; seed 301",
            fontsize=10,
        )
        name = f"{family}_g{g}_s{s}_corr{corr}_scale{scale}"
        fig.savefig(figdir / (name + ".pdf"))
        fig.savefig(figdir / (name + ".png"), dpi=150)
        plt.close(fig)
    (figdir / "READ_ME.txt").write_text(
        "Left: solid measured forward energy error; dashed theoretical recurrence evaluated in FP64; dotted residual-gap baseline. Middle: bound/error; gaps mean unavailable bound or zero denominator. Right: solid distance to FP64 preconditioner output on the SAME input, dashed eta+eta_ref; this is not the exact call error. All plots use fixed seed 301; summary includes every seed and every failure. GPU evaluations are not outward-certified.\n"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("directory", type=Path)
    a = p.parse_args()
    summarize(a.directory)
