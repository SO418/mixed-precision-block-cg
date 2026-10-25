import sys, json, math, shutil
from pathlib import Path
from fractions import Fraction as F
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import LogLocator, NullLocator, MaxNLocator

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE.parent / "exact"))
from certify_budgets import fm, reference_p, solve_exact, eye, ss, root as exact_sqrt

root = BASE / "results" / "exact"
out = BASE / "results" / "figures"
out.mkdir(exist_ok=True, parents=True)
plt.rcParams.update(
    {
        "font.family": "STIXGeneral",
        "mathtext.fontset": "stix",
        "font.size": 12,
        "axes.labelsize": 14,
        "axes.titlesize": 14,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.7,
        "axes.spines.top": True,
        "axes.spines.right": True,
        "savefig.facecolor": "white",
    }
)
colors = {"fp64": "#283E60", "fp32": "#176A8C", "fp16": "#C45A2A", "bf16": "#805A9E"}
markers = {"fp64": "D", "fp32": "s", "fp16": "o", "bf16": "^"}


def ratio(a, b):
    return a / b if b else None


rows = {}
summaries = []
for near in [0, 1]:
    for fmt in ["fp64", "fp32", "fp16", "bf16"]:
        name = f"n16_s4_{fmt}_near{near}_scale0_steps50"
        src = root
        t = json.loads((src / (name + "_trace.json")).read_text())
        c = json.loads((src / (name + "_certificate.json")).read_text())
        v = json.loads((src / (name + "_result.json")).read_text())["validation"]
        a = fm(t["spec"]["A"])
        B = fm(t["spec"]["B"])
        P = reference_p(a, t["spec"])
        Pi = solve_exact(P, eye(16))
        ap = F(c["constants"]["aplus"])
        chi = F(c["constants"]["chi"])
        d = []
        for tr, cs, vs in zip(t["steps"], c["steps"], v["steps"]):
            R = fm(tr["R"])
            Z = fm(tr["Z"])
            X = fm(tr["Xnext"])
            be2 = ss(Pi @ Z - R)
            ce2 = ss(B - a @ X)
            eta = F(cs["eta"])
            U = F(cs["Unext"])
            assert (
                F(vs["actual_eta_squared"]) <= eta * eta
                and F(vs["actual_next_energy_squared"]) <= U * U
            )
            assert be2 <= ap * eta * eta / (1 - chi) ** 2
            assert ce2 <= ap * U * U
            d.append(
                dict(
                    k=tr["j"],
                    b_f=float(exact_sqrt(F(vs["actual_eta_squared"]))),
                    b_fb=float(eta),
                    b_b=float(exact_sqrt(be2)),
                    b_bb=math.sqrt(float(ap)) * float(eta / (1 - chi)),
                    c_f=float(exact_sqrt(F(vs["actual_next_energy_squared"]))),
                    c_fb=float(U),
                    c_b=float(exact_sqrt(ce2)),
                    c_bb=math.sqrt(float(ap)) * float(U),
                )
            )

        def floor(start):
            h = c["steps"][start:]
            dd = max(float(F(q["zeta"])) for q in h)
            w = max(float(F(q["omega"])) for q in h)
            xx = max(float(F(q["scalars"]["x"])) for q in h)
            ch = float(chi)
            cc = dd + ch * xx
            return xx + (ch * cc + math.hypot(cc, math.sqrt(1 - ch * ch) * math.sqrt(w))) / (
                1 - ch * ch
            )

        fl = floor(25)
        fullfl = floor(0)
        row = dict(
            data=d,
            tailF=fl,
            fullF=fullfl,
            aplus=float(ap),
            chi=float(chi),
            U25=float(F(c["steps"][25]["U"])),
            source=str(src / name),
        )
        rows[near, fmt] = row
        summaries.append(
            dict(
                near=near,
                precision=fmt,
                BJAC_forward_ratio=ratio(d[-1]["b_fb"], d[-1]["b_f"]),
                BJAC_backward_ratio=ratio(d[-1]["b_bb"], d[-1]["b_b"]),
                CG_forward_ratio=ratio(d[-1]["c_fb"], d[-1]["c_f"]),
                CG_backward_ratio=ratio(d[-1]["c_bb"], d[-1]["c_b"]),
                tailF=fl,
                fullF=fullfl,
            )
        )
        print("loaded", near, fmt, flush=True)
with PdfPages(out / "figures.pdf") as pdf:
    for method in ["BJAC", "BlockCG"]:
        for near in [0, 1]:
            fig, axs = plt.subplots(1, 2, figsize=(11.4, 4.65))
            fig.subplots_adjust(left=0.09, right=0.985, top=0.73, bottom=0.17, wspace=0.27)
            title = (
                ("BJAC application" if method == "BJAC" else "BJAC-preconditioned block CG")
                + "  |  "
                + ("Correlated right-hand sides" if near else "Independent right-hand sides")
            )
            fig.suptitle(title, x=0.535, y=0.977, fontsize=16)
            fig.text(
                0.535,
                0.904,
                r"$n=16,\ s=4,\ b=4,\ t=3,\ p=2;\quad u_{\mathrm{outer}}=u_{64}$",
                ha="center",
                fontsize=12,
            )
            legend = [
                Line2D(
                    [0],
                    [0],
                    color=colors[f],
                    marker=markers[f],
                    mfc="white",
                    lw=1.8,
                    label=f.upper(),
                )
                for f in ["fp64", "fp32", "fp16", "bf16"]
            ]
            legend += [
                Line2D([0], [0], color="#333333", lw=1.8, label="Measured error"),
                Line2D([0], [0], color="#333333", lw=1.8, ls=(0, (5, 3)), label="Stepwise bound"),
            ]
            if method == "BlockCG":
                legend.append(
                    Line2D(
                        [0],
                        [0],
                        color="#333333",
                        lw=2.3,
                        ls=(0, (1, 2)),
                        label="Interval accuracy bound",
                    )
                )
            fig.legend(
                handles=legend,
                loc="upper center",
                bbox_to_anchor=(0.535, 0.858),
                ncol=len(legend),
                frameon=False,
                fontsize=9.5,
                columnspacing=0.85,
                handlelength=2.1,
            )
            for j, ax in enumerate(axs):
                typ = ("b_" if method == "BJAC" else "c_") + ("f" if j == 0 else "b")
                for fmt in ["fp64", "fp32", "fp16", "bf16"]:
                    r = rows[near, fmt]
                    d = r["data"]
                    k = [q["k"] + (method == "BlockCG") for q in d]
                    v = [q[typ] for q in d]
                    u = [q[typ + "b"] for q in d]
                    ax.semilogy(
                        k,
                        np.ma.masked_less_equal(v, 0),
                        color=colors[fmt],
                        marker=markers[fmt],
                        mfc="white",
                        mew=0.9,
                        markersize=4,
                        markevery=[0, 2, 5, 10, 20, 30, 40, 49],
                        zorder=3,
                    )
                    ax.semilogy(k, u, color=colors[fmt], ls=(0, (5, 3)), lw=1.7, zorder=2)
                    if method == "BlockCG":
                        # Finite interval includes remaining transient; F alone is not substituted for the bound.
                        kk = np.arange(26, 51)
                        ff = r["tailF"] + r["chi"] ** (kk - 25) * max(r["U25"] - r["tailF"], 0)
                        if j:
                            ff = ff * math.sqrt(r["aplus"])
                        ax.semilogy(kk, ff, color=colors[fmt], ls=(0, (1, 2)), lw=2.5, zorder=4)
                ax.set_title("(a) Forward error" if j == 0 else "(b) Backward error", pad=10)
                if method == "BJAC":
                    ax.set_ylabel(
                        r"$\|\widehat Z_k-PR_k\|_{A,F}$"
                        if j == 0
                        else r"$\|P^{-1}\widehat Z_k-R_k\|_F$"
                    )
                else:
                    ax.set_ylabel(
                        r"$\|X_*-\widehat X_k\|_{A,F}$" if j == 0 else r"$\|B-A\widehat X_k\|_F$"
                    )
                ax.set_xlabel("Preconditioner call $k$" if method == "BJAC" else "Iteration $k$")
                ax.set_xlim(0, 50)
                ax.set_xticks([0, 10, 20, 30, 40, 50])
                ax.yaxis.set_major_locator(LogLocator(base=10, numticks=6))
                ax.yaxis.set_minor_locator(NullLocator())
                ax.grid(axis="y", color="#e1e4e8", lw=0.6)
                ax.tick_params(direction="in", top=True, right=True, length=4)
            number = (1 if method == "BJAC" else 3) + near
            name = f"Fig{number}_{method}_" + ("correlated" if near else "independent")
            fig.savefig(out / (name + ".png"), dpi=240)
            fig.savefig(out / (name + ".pdf"))
            pdf.savefig(fig)
            plt.close(fig)
(out / "metrics.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2))
(out / "plot_data.json").write_text(
    json.dumps({str(k): v for k, v in rows.items()}, ensure_ascii=False)
)
print(json.dumps(summaries, ensure_ascii=False))
