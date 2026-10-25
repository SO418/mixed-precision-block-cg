#!/usr/bin/env python3
"""Computable bounds for BJAC and the outer iteration, checked in exact rational arithmetic.

The monitor builds the bounds from the stored trajectory, primitive rounding bounds and
analytic spectral estimates. The validator compares them with exact errors obtained from
P and A^{-1}, which the monitor never uses. Supported matrices are symmetric with unit
diagonal and exactly representable in the inner format.
"""

import argparse
import hashlib
import json
import math
import platform
from fractions import Fraction as F
from pathlib import Path

import numpy as np
from formats import quantize, conversion_model

from verify_qr_bjac import Arithmetic, load_record

HERE = Path(__file__).resolve().parent
BITS = 128
UH, TH = F(1, 2**53), F(1, 2**1075)


def two(e):
    return F(2**e) if e >= 0 else F(1, 2 ** (-e))


def log2floor(x):
    assert x > 0
    e = x.numerator.bit_length() - x.denominator.bit_length()
    return e - int(x < two(e))


def up(x):
    """Round a nonnegative rational upwards to <= BITS binary digits."""
    x = F(x)
    assert x >= 0
    if not x:
        return F(0)
    step = two(log2floor(x) - BITS + 1)
    y = x / step
    ans = ((y.numerator + y.denominator - 1) // y.denominator) * step
    assert ans >= x
    return ans


def sqrt_interval(x):
    """Integer square root gives an exact rational outward enclosure."""
    x = F(x)
    assert x >= 0
    if not x:
        return F(0), F(0)
    step = two(log2floor(x) // 2 - BITS + 1)
    y = x / (step * step)
    m = math.isqrt(y.numerator // y.denominator)
    lo = m * step
    hi = lo if m * m * y.denominator == y.numerator else (m + 1) * step
    assert lo * lo <= x <= hi * hi
    return lo, hi


def root(x):
    return sqrt_interval(x)[1]


def fm(a):
    a = np.asarray(a)
    return np.array([F(float(v)) for v in a.flat], dtype=object).reshape(a.shape)


def zeros(shape):
    return np.full(shape, F(0), dtype=object)


def eye(n):
    a = zeros((n, n))
    for j in range(n):
        a[j, j] = F(1)
    return a


def fmap(fn, a):
    return np.array([fn(v) for v in a.flat], dtype=object).reshape(a.shape)


def ss(a):
    return sum((v * v for v in a.flat), F(0))


def fnorm(a):
    return root(ss(a))


def anorm_bound(a, aplus):
    return root(aplus * ss(a))


def energy2(a, x):
    ax = a @ x
    return sum((v * w for v, w in zip(x.flat, ax.flat)), F(0))


def solve_exact(a, b):
    """Rational Gaussian elimination (validator only)."""
    a, b = a.copy(), b.copy()
    n = a.shape[0]
    for k in range(n):
        pivot = next(j for j in range(k, n) if a[j, k])
        if pivot != k:
            a[[k, pivot]] = a[[pivot, k]]
            b[[k, pivot]] = b[[pivot, k]]
        d = a[k, k]
        a[k, k:] /= d
        b[k, :] /= d
        for j in range(n):
            if j == k:
                continue
            d = a[j, k]
            if d:
                a[j, k:] -= d * a[k, k:]
                b[j, :] -= d * b[k, :]
    return b


def menv(a, b, u=UH, tau=TH):
    q = a.shape[1]
    if not q:
        return zeros((a.shape[0], b.shape[1]))
    aq = (1 + u) ** (q + 1) - 1
    bq = tau * (2 + u) * sum(((1 + u) ** j for j in range(q)), F(0))
    return fmap(up, aq * (abs(a) @ abs(b)) + bq)


def aenv(a, b, u=UH, tau=TH):
    return fmap(up, u * (abs(a) + abs(b)) + tau)


class LowMonitor:
    """Primitive rounding bounds computed from stored outputs."""

    def __init__(self, precision):
        self.precision = precision
        self.ul, self.tl = conversion_model(precision)
        # Explicit NumPy simulation computes in fp64 before its low conversion.
        self.u = UH + self.ul + UH * self.ul
        self.tau = (1 + self.ul) * TH + self.tl
        self.checks = 0

    def rounded(self, value, exact, conversion=False):
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            out = quantize(value, self.precision)
        assert np.all(np.isfinite(out)), "Unsupported overflow/nonfinite result"
        u, tau = (self.ul, self.tl) if conversion else (self.u, self.tau)
        stored = fm(out)
        bound = fmap(lambda v: up((u * abs(v) + tau) / (1 - u)), stored)
        for idx in np.ndindex(stored.shape):
            # Only exact zero operations may discard the absolute term.
            if exact[idx] == 0 and stored[idx] == 0:
                bound[idx] = F(0)
            assert abs(stored[idx] - exact[idx]) <= bound[idx]
            self.checks += 1
        return out, bound

    def cast(self, a):
        return self.rounded(a, fm(a), conversion=True)

    def op(self, a, b, kind):
        a, b = np.asarray(a), np.asarray(b)
        ar, br = fm(a), fm(b)
        if kind == "+":
            return self.rounded(a + b, ar + br)
        if kind == "-":
            return self.rounded(a - b, ar - br)
        if kind == "*":
            return self.rounded(a * b, ar * br)
        if kind == "/":
            assert np.all(br != 0)
            return self.rounded(a / b, ar / br)
        raise ValueError(kind)

    def mm(self, a, b):
        out = np.zeros((a.shape[0], b.shape[1]))
        bound = zeros(out.shape)
        for j in range(a.shape[1]):
            p, ep = self.op(a[:, j : j + 1], b[j : j + 1, :], "*")
            out, ea = self.op(out, p, "+")
            bound = fmap(up, bound + ep + ea)
        assert np.all(abs(fm(out) - fm(a) @ fm(b)) <= bound)
        return out, bound


def family_constants(a, blocksize, sweeps, inner_steps):
    """All returned spectral bounds are rational and computed from stored A."""
    n = a.shape[0]
    assert np.array_equal(a, a.T)
    assert all(a[j, j] == 1 for j in range(n))
    off = a - eye(n)
    radius = max(sum(abs(v) for v in row) for row in off)
    amin, aplus = 1 - radius, 1 + radius
    assert amin > 0
    blockdiag = zeros(a.shape)
    qs = []
    for beg in range(0, n, blocksize):
        sl = slice(beg, min(beg + blocksize, n))
        ab = a[sl, sl]
        blockdiag[sl, sl] = ab
        j = eye(ab.shape[0]) - ab
        qs.append(max(sum(abs(v) for v in row) for row in j))
    q = max(qs)
    assert q < 1
    bt = sum((q**j for j in range(inner_steps)), F(0))
    offblock = max(sum(abs(v) for v in row) for row in (blockdiag - a))
    rho = q**inner_steps + bt * offblock
    assert rho < 1, "This sufficient BJAC spectral certificate is unavailable"
    return dict(
        amin=amin,
        aplus=aplus,
        q=q,
        block_q=qs,
        bt=bt,
        offblock=offblock,
        rho=rho,
        chi=rho**sweeps,
        betab=1 + rho,
    )


def bjac_budget(a_float, r_float, expected_z, stats, spec, const):
    low = LowMonitor(spec["precision"])
    alow, _ = low.cast(a_float)
    assert np.array_equal(alow, a_float), "Matrix conversion needs an extra budget"
    n, width = r_float.shape
    output = np.zeros_like(r_float)
    column_bounds, details = [], []
    t, sweeps, bs = spec["t"], spec["k"], spec["blocksize"]
    for col in range(width):
        rhs, cast = low.cast(r_float[:, col : col + 1])
        ccast = up((1 + const["chi"]) * root(ss(cast) / const["amin"]))
        y = np.zeros((n, 1))
        propagated = F(0)
        sweep_detail = []
        for sw in range(sweeps):
            ay, ew = low.mm(alow, y)
            residual, er = low.op(rhs, ay, "-")
            eres = fmap(up, ew + er)
            correction = np.zeros_like(y)
            lis = []
            for block, beg in enumerate(range(0, n, bs)):
                sl = slice(beg, min(beg + bs, n))
                ab = alow[sl, sl]
                u = np.zeros((ab.shape[0], 1))
                local = F(0)
                for _ in range(t):
                    w, wg = low.mm(ab, u)
                    rr, rv = low.op(residual[sl], w, "-")
                    du, rc = low.op(rr, np.diag(ab)[:, None], "/")
                    u, ra = low.op(u, du, "+")
                    dq = fmap(up, wg + rv + rc + ra)  # D=R=I, reciprocal defect is zero.
                    local = up(const["block_q"][block] * local + fnorm(dq))
                qb = const["block_q"][block]
                smin = (1 - qb**t) / (1 + qb)
                lis.append(up(local * root(1 / smin)))
                correction[sl] = u
            ynext, eupd = low.op(y, correction, "+")
            cr = root(const["betab"] * const["bt"] * ss(eres))
            ci = root(const["betab"] * sum((li * li for li in lis), F(0)))
            cu = anorm_bound(eupd, const["aplus"])
            cj = up(cr + ci + cu)
            propagated = up(const["rho"] * propagated + cj)
            saved = stats["column_sweeps"][col][sw]
            assert np.array_equal(residual, np.asarray(saved["residual"]))
            assert np.array_equal(ynext, np.asarray(saved["z"]))
            sweep_detail.append(
                dict(cj=cj, residual_part=cr, block_part=ci, update_part=cu, block_L=lis)
            )
            y = ynext
        output[:, col : col + 1] = y
        column_bounds.append(up(ccast + propagated))
        details.append(dict(cast=ccast, convolution=propagated, sweeps=sweep_detail))
    assert np.array_equal(output, expected_z), "BJAC replay does not match the saved algorithm"
    return dict(
        eta=root(sum((v * v for v in column_bounds), F(0))),
        column_eta=column_bounds,
        columns=details,
        primitive_checks=low.checks,
    )


def recomputed_residual_envelope(a, b, target):
    """Enclose a*b-target, using fp64 monitor products and explicit envelopes."""
    arith = Arithmetic("fp64")
    product = arith.mm(np.asarray(a, dtype=float), np.asarray(b, dtype=float))
    residual = arith.q(product - np.asarray(target, dtype=float))
    bound = fmap(up, abs(fm(residual)) + menv(a, b) + aenv(fm(product), -target))
    assert np.all(abs(a @ b - target) <= bound)
    return bound


def check_trajectory(record):
    """Check the saved operation path and continuity before building a bound."""
    spec, initial = record["spec"], record["initial"]
    arith = Arithmetic("fp64")

    def same(a, b):
        assert np.array_equal(a, b), "Stored operation path or step continuity mismatch"

    same(spec["X0"], initial["X"])
    same(initial["R"], arith.q(spec["B"] - arith.mm(spec["A"], initial["X"])))
    for j, row in enumerate(record["steps"]):
        assert row["j"] == j
        for key in ("X", "R", "Z", "Q", "T"):
            same(row[key], initial[key] if j == 0 else record["steps"][j - 1][key + "next"])
        if j == 0:
            assert all(row[key] is None for key in ("previous_Q", "previous_beta", "previous_V"))
        else:
            for key, old in (("previous_Q", "Q"), ("previous_beta", "beta"), ("previous_V", "V")):
                same(row[key], record["steps"][j - 1][old])
        for result, left, right in (
            ("W", spec["A"], row["Q"]),
            ("C", row["Q"].T, row["W"]),
            ("D", row["Q"].T, row["R"]),
            ("Qalpha", row["Q"], row["alpha"]),
            ("Walpha", row["W"], row["alpha"]),
            ("F", row["W"].T, row["Znext"]),
            ("Qbeta", row["Q"], row["beta"]),
        ):
            same(row[result], arith.mm(left, right))
        same(row["Xnext"], arith.q(row["X"] + row["Qalpha"]))
        same(row["Rnext"], arith.q(row["R"] - row["Walpha"]))
        same(row["V"], arith.q(row["Znext"] + row["Qbeta"]))


def monitor(record):
    check_trajectory(record)
    spec = record["spec"]
    a, b = fm(spec["A"]), fm(spec["B"])
    const = family_constants(a, spec["blocksize"], spec["k"], spec["t"])
    lowa, higha, chi = const["amin"], const["aplus"], const["chi"]
    x0, r0 = fm(record["initial"]["X"]), fm(record["initial"]["R"])
    ax0 = Arithmetic("fp64").mm(spec["A"], record["initial"]["X"])
    g0env = fmap(up, menv(a, x0) + aenv(b, -fm(ax0)))
    gamma = root(ss(g0env) / lowa)
    U = up(root(ss(r0) / lowa) + gamma)
    rows, previous = [], None
    for row in record["steps"]:
        v = {
            key: fm(row[key])
            for key in (
                "X",
                "R",
                "Z",
                "Q",
                "T",
                "W",
                "C",
                "D",
                "alpha",
                "Qalpha",
                "Walpha",
                "Xnext",
                "Rnext",
                "Znext",
                "F",
                "beta",
                "Qbeta",
                "V",
            )
        }
        x, r, z, q, t, w, c, d, alpha, qa, wa, xn, rn, zn, f, beta, qb, vnext = v.values()
        bj = bjac_budget(spec["A"], row["R"], row["Z"], row["bjac_current"], spec, const)
        qrinput = z if previous is None else fm(row["previous_V"])
        eqr = recomputed_residual_envelope(q, t, qrinput)
        orth = recomputed_residual_envelope(q.T, q, eye(q.shape[1]))
        delta = fnorm(orth)
        assert delta < 1
        h = lowa * (1 - delta)  # Lower bound: never round upwards.
        qnorm = root(1 + delta)
        qAnorm = root(higha * (1 + delta))
        env = dict(
            w=menv(a, q),
            c=menv(q.T, w),
            d=menv(q.T, r),
            f=menv(w.T, zn),
            x=fmap(up, menv(q, alpha) + aenv(x, qa)),
            r=fmap(up, menv(w, alpha) + aenv(r, -wa)),
            v=fmap(up, menv(q, beta) + aenv(zn, qb)),
            sa=recomputed_residual_envelope(c, alpha, d),
            sb=recomputed_residual_envelope(c, beta, -f),
            qr=eqr,
        )
        scal = {k: fnorm(val) for k, val in env.items()}
        for k in ("x", "v", "qr"):
            scal[k] = anorm_bound(env[k], higha)
        epsC = up(qnorm * scal["w"] + scal["c"])
        assert epsC < h, "Cannot certify this stored C as nonsingular"
        abar = up((scal["sa"] + scal["d"] + epsC * fnorm(alpha)) * root(1 / h))
        ell = up(scal["qr"] + (previous["v"] if previous else 0))
        bbar, jbar = F(0), F(0)
        if previous:
            jbar = up(
                previous["sb"]
                + previous["f"]
                + previous["w"] * fnorm(z)
                + previous["epsC"] * fnorm(fm(row["previous_beta"]))
                + previous["qA"] * ell
            )
            bbar = up((previous["tbar"] + jbar) * root(1 / previous["h"]))
        tbar = up(scal["sa"] + scal["d"] + scal["c"] * fnorm(alpha) + qnorm * scal["r"])
        zeta = up(bj["eta"] + (1 + chi) * gamma + ell)
        omega = (abar + gamma) ** 2 + ((bbar + gamma) ** 2 if previous else 0)
        Unext = up(root((chi * U + zeta) ** 2 + omega) + scal["x"])
        gnext = up(gamma + (scal["w"] * fnorm(alpha) + scal["r"]) * root(1 / lowa) + scal["x"])
        # Independent residual baseline, not fed into the main recurrence.
        baseline = up(root(ss(rn) / lowa) + gnext)
        cert = dict(
            j=row["j"],
            U=U,
            Unext=Unext,
            gamma=gamma,
            gamma_next=gnext,
            eta=bj["eta"],
            ell=ell,
            a=abar,
            b=bbar,
            zeta=zeta,
            omega=up(omega),
            delta=delta,
            h=h,
            epsC=epsC,
            tbar=tbar,
            jbar=jbar,
            baseline_next=baseline,
            scalars=scal,
            bjac=bj,
        )
        rows.append(cert)
        previous = dict(scal, h=h, epsC=epsC, tbar=tbar, qA=qAnorm)
        U, gamma = Unext, gnext
    return dict(
        name=spec["name"],
        constants=const,
        initial_U=rows[0]["U"],
        initial_gamma=rows[0]["gamma"],
        steps=rows,
    )


def reference_p(a, spec):
    """Exact two-level BJAC polynomial (validator only)."""
    n, bs = a.shape[0], spec["blocksize"]
    bt = zeros(a.shape)
    for beg in range(0, n, bs):
        sl = slice(beg, min(beg + bs, n))
        ab = a[sl, sl]
        j = eye(ab.shape[0]) - ab
        power = eye(ab.shape[0])
        for _ in range(spec["t"]):
            bt[sl, sl] += power
            power = power @ j
    h = eye(n) - bt @ a
    p, power = zeros(a.shape), eye(n)
    for _ in range(spec["k"]):
        p += power @ bt
        power = power @ h
    return p


def validate(record, cert):
    """Compare squared bounds and exact errors without tolerances."""
    spec = record["spec"]
    a, b = fm(spec["A"]), fm(spec["B"])
    ainv = solve_exact(a, eye(a.shape[0]))
    xs = ainv @ b
    p = reference_p(a, spec)
    e02 = energy2(a, xs - fm(record["initial"]["X"]))
    comparisons = 0
    rows = []
    for row, bound in zip(record["steps"], cert["steps"]):
        x, r, z, q, t, w, c, d, alpha, xn, rn, zn, f, beta, v = (
            fm(row[k])
            for k in (
                "X",
                "R",
                "Z",
                "Q",
                "T",
                "W",
                "C",
                "D",
                "alpha",
                "Xnext",
                "Rnext",
                "Znext",
                "F",
                "beta",
                "V",
            )
        )
        ew, ec, ed = w - a @ q, c - q.T @ w, d - q.T @ r
        ex, er, ef, ev = xn - x - q @ alpha, rn - r + w @ alpha, f - w.T @ zn, v - zn - q @ beta
        sa, sb = d - c @ alpha, -f - c @ beta
        g, gn = r - (b - a @ x), rn - (b - a @ xn)
        qrinput = z if row["previous_Q"] is None else fm(row["previous_V"])
        qrdef = q @ t - qrinput
        local = dict(w=ew, c=ec, d=ed, f=ef, x=ex, r=er, v=ev, sa=sa, sb=sb, qr=qrdef)
        checks = {
            "local_"
            + k: (bound["scalars"][k] ** 2, energy2(a, val) if k in ("x", "v", "qr") else ss(val))
            for k, val in local.items()
        }
        h = q.T @ a @ q
        da = q.T @ r - h @ alpha
        ell_actual = qrdef
        if row["previous_Q"] is not None:
            prevq, prevb = fm(row["previous_Q"]), fm(row["previous_beta"])
            ell_actual = q @ t - z - prevq @ prevb
            weighted = prevq.T @ a @ q @ t
            bb = prevq.T @ r - weighted
            checks["b"] = bound["b"] ** 2, energy2(
                solve_exact(prevq.T @ a @ prevq, eye(h.shape[0])), bb
            )
            checks["jbar"] = bound["jbar"] ** 2, ss(weighted)
        checks.update(
            eta=(bound["eta"] ** 2, energy2(a, z - p @ r)),
            gamma=(bound["gamma"] ** 2, energy2(ainv, g)),
            gamma_next=(bound["gamma_next"] ** 2, energy2(ainv, gn)),
            U=(bound["U"] ** 2, energy2(a, xs - x)),
            Unext=(bound["Unext"] ** 2, energy2(a, xs - xn)),
            baseline=(bound["baseline_next"] ** 2, energy2(a, xs - xn)),
            ell=(bound["ell"] ** 2, energy2(a, ell_actual)),
            a=(bound["a"] ** 2, energy2(solve_exact(h, eye(h.shape[0])), da)),
            tbar=(bound["tbar"] ** 2, ss(q.T @ rn)),
            orth=(bound["delta"] ** 2, ss(q.T @ q - eye(q.shape[1]))),
        )
        for label, (upper, actual) in checks.items():
            assert upper >= actual >= 0, (spec["name"], row["j"], label)
            comparisons += 1
        rows.append(
            dict(
                j=row["j"],
                exact_squared_comparisons=len(checks),
                actual_next_energy_squared=energy2(a, xs - xn),
                actual_eta_squared=energy2(a, z - p @ r),
                actual_gap_squared=energy2(ainv, g),
                actual_squared={key: pair[1] for key, pair in checks.items()},
            )
        )
    final = cert["steps"][-1]
    ratio = root(rows[-1]["actual_next_energy_squared"] / e02)
    return dict(
        exact_squared_comparisons=comparisons,
        initial_error_squared=e02,
        true_relative_error_upper=ratio,
        certified_U_over_initial_error_upper=root(final["Unext"] ** 2 / e02),
        baseline_over_initial_error_upper=root(final["baseline_next"] ** 2 / e02),
        steps=rows,
    )


def serial(obj):
    if isinstance(obj, F):
        return str(obj.numerator) + "/" + str(obj.denominator)
    if isinstance(obj, dict):
        return {k: serial(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [serial(v) for v in obj]
    return obj


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=HERE / "input_traces")
    parser.add_argument("--output", type=Path, default=HERE / "certified_results")
    parser.add_argument("--case")
    args = parser.parse_args()
    paths = sorted(args.input.glob("*_trace.json"))
    if args.case:
        paths = [p for p in paths if p.name == args.case + "_trace.json"]
    assert paths
    args.output.mkdir(parents=True, exist_ok=True)
    report = dict(
        status="exact rational checks of stored traces",
        arithmetic="upward rounding to 128 significant bits; integer square-root enclosures",
        scope="D=I; symmetric strictly diagonally dominant A, exactly representable in the inner format; no overflow",
        python=platform.python_version(),
        numpy=np.__version__,
        cases=[],
    )
    for path in paths:
        record = load_record(path)
        cert = monitor(record)
        validation = validate(record, cert)
        name = cert["name"]
        output = args.output / (name + "_certificate.json")
        output.write_text(
            json.dumps(serial(dict(certificate=cert, validation=validation)), indent=2) + "\n"
        )
        item = dict(
            name=name,
            steps=len(cert["steps"]),
            primitive_checks=sum(x["bjac"]["primitive_checks"] for x in cert["steps"]),
            exact_squared_comparisons=validation["exact_squared_comparisons"],
            actual_error_ratio_display=float(validation["true_relative_error_upper"]),
            U_ratio_display=float(validation["certified_U_over_initial_error_upper"]),
            baseline_ratio_display=float(validation["baseline_over_initial_error_upper"]),
            max_eta_ratio_display=max(
                float(root(c["eta"] ** 2 / v["actual_eta_squared"]))
                for c, v in zip(cert["steps"], validation["steps"])
                if v["actual_eta_squared"]
            ),
            input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            certificate_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
        )
        report["cases"].append(item)
        print(
            name,
            "actual={:.3e} U={:.3e}".format(
                item["actual_error_ratio_display"], item["U_ratio_display"]
            ),
            flush=True,
        )
    report["source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = [
        "# Exact rational check of the computable bounds",
        "",
        "Decimals are for display; the certificate files store exact rational bounds.",
        "",
        "| case | error / initial error | U / initial error | residual-gap bound / initial error |",
        "|---|---:|---:|---:|",
    ]
    for c in report["cases"]:
        lines.append(
            "| {} | {:.6e} | {:.6e} | {:.6e} |".format(
                c["name"],
                c["actual_error_ratio_display"],
                c["U_ratio_display"],
                c["baseline_ratio_display"],
            )
        )
    (args.output / "SUMMARY.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
