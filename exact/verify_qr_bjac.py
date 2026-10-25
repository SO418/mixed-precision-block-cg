#!/usr/bin/env python3
"""Rounded BJAC preconditioning with QR-based block CG, and high-precision checks.

Run: python3 verify_qr_bjac.py   (numpy 2.0.2, mpmath 1.3.0)
Reductions use separate multiplies and adds in ascending index order. QR and the small
dense solves come from LAPACK in FP64; other LAPACK builds may change the last bits.
They are checked through their stored outputs.
"""

import argparse
import hashlib
import json
import platform
from fractions import Fraction
from pathlib import Path

import mpmath as mp
import numpy as np
from formats import quantize

HERE = Path(__file__).resolve().parent
mp.mp.dps = 100


class Arithmetic:
    def __init__(self, precision):
        self.precision = precision
        self.dtype = None
        self.underflow_to_zero = 0
        self.rounded_values = 0

    def q(self, a):
        a = np.asarray(a, dtype=np.float64)
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            b = quantize(a, self.precision)
        self.underflow_to_zero += int(np.count_nonzero((a != 0) & (b == 0)))
        self.rounded_values += int(np.count_nonzero(a != b))
        return b

    def mm(self, a, b):
        a, b = np.asarray(a), np.asarray(b)
        c = np.zeros((a.shape[0], b.shape[1]))
        for k in range(a.shape[1]):
            c = self.q(c + self.q(a[:, k : k + 1] * b[k : k + 1, :]))
        return c


OUTER = Arithmetic("fp64")


def bjac(a, r, blocksize, k, t, precision):
    """k residual-correction sweeps; each block is solved by t Jacobi steps from zero.

    Coefficients are rounded to the inner format once per call, and residuals and
    inner vectors are kept in that format. Columns are processed independently.
    """
    low = Arithmetic(precision)
    alow = low.q(a)
    n, s = r.shape
    z = np.zeros_like(r)
    stagnated_sweeps = 0
    unchanged_sweeps = 0
    histories = []
    for col in range(s):
        rhs = low.q(r[:, col : col + 1])
        y = np.zeros((n, 1))
        column_history = []
        for _ in range(k):
            residual = low.q(rhs - low.mm(alow, y))
            correction = np.zeros_like(y)
            for begin in range(0, n, blocksize):
                sl = slice(begin, begin + blocksize)
                ab = alow[sl, sl]
                u = np.zeros((ab.shape[0], 1))
                d = np.diag(ab)[:, None]
                for _ in range(t):
                    w = low.mm(ab, u)
                    rr = low.q(residual[sl] - w)
                    du = low.q(rr / d)
                    u = low.q(u + du)
                correction[sl] = u
            ynext = low.q(y + correction)
            unchanged = bool(np.array_equal(ynext, y))
            unchanged_sweeps += int(unchanged)
            if unchanged and np.any(residual != 0):
                stagnated_sweeps += 1
            column_history.append(
                {"z": ynext.tolist(), "residual": residual.tolist(), "unchanged": unchanged}
            )
            y = ynext
        z[:, col : col + 1] = y
        histories.append(column_history)
    return z, {
        "underflow_to_zero": low.underflow_to_zero,
        "rounded_values": low.rounded_values,
        "unchanged_sweeps": unchanged_sweeps,
        "stagnated_sweeps_with_nonzero_recursive_inner_residual": stagnated_sweeps,
        "column_sweeps": histories,
    }


def qr_raw(v):
    q, t = np.linalg.qr(v, mode="reduced")
    sv = np.linalg.svd(v, compute_uv=False)
    # Numerical rank is only reported; the full Q from LAPACK is used even if V is singular.
    threshold = max(v.shape) * np.finfo(float).eps * (sv[0] if len(sv) else 0)
    return (
        q,
        t,
        {
            "singular_values_fp64": sv.tolist(),
            "rank_estimate_fp64": int(np.count_nonzero(sv > threshold)),
            "rank_threshold_fp64": float(threshold),
        },
    )


def make_case(
    name,
    s,
    precision,
    rhs_mode="random",
    nonzero_x=False,
    scale=1.0,
    k=2,
    t=3,
    iterations=18,
    seed=20260911,
):
    n = 12
    # Symmetric strict diagonal dominance gives SPD; Gershgorin gives
    # spec(A) subset [7/8,9/8]. D=I and ||I-A||inf<=1/8 prove convergence.
    a = np.eye(n)
    for i in range(n - 1):
        a[i, i + 1] = a[i + 1, i] = -1 / 16
    rng = np.random.default_rng(seed)
    b = rng.standard_normal((n, s))
    if rhs_mode in ("near", "exact"):
        base = b[:, :1].copy()
        for j in range(1, s):
            b[:, j : j + 1] = base if rhs_mode == "exact" else base + 1e-11 * b[:, j : j + 1]
    b *= scale
    x = rng.standard_normal((n, s)) / 8 if nonzero_x else np.zeros((n, s))
    return dict(
        name=name,
        A=a,
        B=b,
        X0=x,
        n=n,
        s=s,
        blocksize=3,
        precision=precision,
        rhs_mode=rhs_mode,
        k=k,
        t=t,
        iterations=iterations,
        seed=seed,
    )


def run_case(spec):
    a, b = spec["A"], spec["B"]
    x = spec["X0"].copy()
    r = OUTER.q(b - OUTER.mm(a, x))
    z, zstats = bjac(a, r, spec["blocksize"], spec["k"], spec["t"], spec["precision"])
    q, t, qrinfo = qr_raw(z)
    initial = dict(
        X=x.copy(), R=r.copy(), Z=z.copy(), Q=q.copy(), T=t.copy(), bjac=zstats, qr=qrinfo
    )
    steps = []
    previous_q = previous_beta = None
    previous_v = None
    for j in range(spec["iterations"]):
        w = OUTER.mm(a, q)
        c = OUTER.mm(q.T, w)
        d = OUTER.mm(q.T, r)
        alpha = np.linalg.solve(c, d)
        qalpha = OUTER.mm(q, alpha)
        xnext = OUTER.q(x + qalpha)
        walpha = OUTER.mm(w, alpha)
        rnext = OUTER.q(r - walpha)
        znext, znextstats = bjac(
            a, rnext, spec["blocksize"], spec["k"], spec["t"], spec["precision"]
        )
        # W and C from this step remain available through the beta solve.
        f = OUTER.mm(w.T, znext)
        beta = -np.linalg.solve(c, f)
        qbeta = OUTER.mm(q, beta)
        v = OUTER.q(znext + qbeta)
        qnext, tnext, qrnextinfo = qr_raw(v)
        row = dict(
            j=j,
            X=x.copy(),
            R=r.copy(),
            Z=z.copy(),
            Q=q.copy(),
            T=t.copy(),
            W=w,
            C=c,
            D=d,
            alpha=alpha,
            Qalpha=qalpha,
            Walpha=walpha,
            Xnext=xnext,
            Rnext=rnext,
            Znext=znext,
            F=f,
            beta=beta,
            Qbeta=qbeta,
            V=v,
            Qnext=qnext,
            Tnext=tnext,
            bjac_current=zstats,
            bjac_next=znextstats,
            qr_current=qrinfo,
            qr_next=qrnextinfo,
            previous_Q=previous_q,
            previous_beta=previous_beta,
            previous_V=previous_v,
        )
        steps.append(row)
        previous_q, previous_beta, previous_v = q.copy(), beta.copy(), v.copy()
        x, r, z, q, t = xnext, rnext, znext, qnext, tnext
        zstats, qrinfo = znextstats, qrnextinfo
    return dict(spec=spec, initial=initial, steps=steps)


def M(a):
    """Each JSON/NumPy binary64 number is interpreted as its exact real value."""
    if a is None:
        return None
    a = np.asarray(a)
    return mp.matrix([[mp.mpf(float(v)) for v in row] for row in a])


def inner(x, y):
    return mp.fsum(x[i, j] * y[i, j] for i in range(x.rows) for j in range(x.cols))


def fro(x):
    return mp.sqrt(max(mp.mpf(0), inner(x, x)))


def solve(a, b):
    out = mp.matrix(a.rows, b.cols)
    for j in range(b.cols):
        out[:, j] = mp.lu_solve(a, b[:, j])
    return out


def energy(a, x):
    return mp.sqrt(max(mp.mpf(0), inner(x, a * x)))


def invenergy(a, x):
    return mp.sqrt(max(mp.mpf(0), inner(x, solve(a, x))))


def norm2(x):
    return mp.sqrt(max(mp.mpf(0), max(mp.eigsy(x.T * x, eigvals_only=True))))


def reference_preconditioner(a, blocksize, k, t):
    n = a.rows
    bt = mp.matrix(n)
    for begin in range(0, n, blocksize):
        ab = a[begin : begin + blocksize, begin : begin + blocksize]
        gg = mp.eye(blocksize) - ab  # D=I in the stated test matrix.
        power = mp.eye(blocksize)
        bb = mp.matrix(blocksize)
        for _ in range(t):
            bb += power
            power = power * gg
        bt[begin : begin + blocksize, begin : begin + blocksize] = bb
    hh = mp.eye(n) - bt * a
    pp = mp.matrix(n)
    power = mp.eye(n)
    for _ in range(k):
        pp += power * bt
        power = power * hh
    # Exact analytic bound: H=G_in^t+B_t E_out, all matrices nonnegative,
    # ||G_in||inf,||E_out||inf<=1/8. A^{1/2} H A^{-1/2} is symmetric.
    rr = Fraction(1, 8)
    chi_rat = (rr**t + (1 - rr**t) / 7) ** k
    chi = mp.mpf(chi_rat.numerator) / chi_rat.denominator
    return pp, chi, str(chi_rat)


def scaled_defect(lhs, rhs, scale=None):
    if isinstance(lhs, mp.matrix):
        err = fro(lhs - rhs)
        natural = fro(lhs) + fro(rhs)
    else:
        err = abs(lhs - rhs)
        natural = abs(lhs) + abs(rhs)
    return err / max(natural if scale is None else scale, mp.mpf("1e-200"))


def number(x):
    return mp.nstr(x, 45)


def exact_q_rank_check(q):
    """Finite rational check: ||Q^T Q-I||_F^2<1/4 implies full column rank.

    This certifies this stored Q's rank only, not the floating-point theorem
    budgets. Fraction.from_float uses exact binary64 values.
    """
    qrat = [[Fraction.from_float(float(x)) for x in row] for row in q]
    s = q.shape[1]
    sq = Fraction(0)
    for i in range(s):
        for j in range(s):
            val = sum(row[i] * row[j] for row in qrat) - int(i == j)
            sq += val * val
    return bool(sq < Fraction(1, 4))


def audit_case(record):
    a, b = M(record["spec"]["A"]), M(record["spec"]["B"])
    xstar = solve(a, b)  # A^{-1} B for the stored input.
    initial_error = energy(a, xstar - M(record["initial"]["X"]))
    ell_a, a_plus = mp.mpf(7) / 8, mp.mpf(9) / 8
    pref, chi, chi_rational = reference_preconditioner(
        a, record["spec"]["blocksize"], record["spec"]["k"], record["spec"]["t"]
    )
    r0, x0 = M(record["initial"]["R"]), M(record["initial"]["X"])
    gbar = invenergy(a, r0 - (b - a * x0))
    ubound = fro(r0) / mp.sqrt(ell_a) + gbar
    previous_defects = None
    reference_defect = fro(b - a * xstar) / max(fro(b), mp.mpf("1e-200"))
    with mp.workdps(160):
        xstar160 = solve(a, b)
        reference_comparison = energy(a, xstar160 - xstar) / energy(a, xstar160)
    results = []
    stagnation_with_nonzero_true_inner_residual = 0
    for row in record["steps"]:
        assert exact_q_rank_check(row["Q"])
        vals = {
            key: M(row[key])
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
                "Xnext",
                "Rnext",
                "Znext",
                "F",
                "beta",
                "V",
                "Qnext",
                "Tnext",
                "previous_Q",
                "previous_beta",
                "previous_V",
            )
        }
        x, r, z, q, t, w, c, d, alpha, xn, rn, zn, f, beta, v, qn, tn, u, bold, vold = vals.values()
        cast_rhs = M(quantize(row["R"], record["spec"]["precision"]))
        for col, column_history in enumerate(row["bjac_current"]["column_sweeps"]):
            for sweep in column_history:
                if sweep["unchanged"] and fro(cast_rhs[:, col] - a * M(sweep["z"])) > 0:
                    stagnation_with_nonzero_true_inner_residual += 1
        exact_r = b - a * x
        g, gn = r - exact_r, rn - (b - a * xn)
        ew, ec, ed = w - a * q, c - q.T * w, d - q.T * r
        ex, er = xn - x - q * alpha, rn - r + w * alpha
        ef, ev, eqr = f - w.T * zn, v - zn - q * beta, qn * tn - v
        ka, kb = d - c * alpha, -f - c * beta
        h, e = q.T * a * q, xstar - x
        da = q.T * r - h * alpha
        l = q * t - z if u is None else q * t - z - u * bold
        current_l_parts = q * t - z if u is None else (vold - z - u * bold) + (q * t - vold)
        kk = t - alpha
        central = energy(a, e - z - l) ** 2
        first_inner = 2 * inner(kk, da - q.T * g)
        u_norm_squared = previous_inner = previous_term = mp.mpf(0)
        previous_projection_residual = None
        if u is not None:
            hu = u.T * a * u
            cw = u.T * a * q * t
            rprev = u.T * (r - g)
            u_norm_squared = energy(a, u * bold) ** 2
            previous_inner = -2 * inner(bold, rprev - cw)
            previous_projection_residual = u.T * r - cw
            previous_term = (invenergy(hu, previous_projection_residual) + invenergy(a, g)) ** 2
        theorem_lhs = energy(a, e - q * alpha) ** 2
        theorem_rhs = (
            central - energy(a, q * kk) ** 2 - u_norm_squared + first_inner + previous_inner
        )
        # Use the scale of all expanded terms, not a cancellation-damaged sum.
        theorem_scale = (
            central
            + energy(a, q * kk) ** 2
            + u_norm_squared
            + abs(first_inner)
            + abs(previous_inner)
            + theorem_lhs
        )
        gapnorm = invenergy(a, g)
        current_term = (invenergy(h, da) + gapnorm) ** 2
        completed_square = central + current_term + previous_term
        stored_bound = mp.sqrt(completed_square) + energy(a, ex)
        actual_next = energy(a, xstar - xn)
        theta = energy(a, solve(a, r) - z) / max(invenergy(a, r), mp.mpf("1e-200"))
        theta_central_bound = (theta * energy(a, e) + (1 + theta) * gapnorm + energy(a, l)) ** 2
        theta_bound = mp.sqrt(theta_central_bound + current_term + previous_term) + energy(a, ex)
        # Check of the recurrence. eta and the local defects are measured in high
        # precision from the stored trajectory (a priori bounds: certify_budgets.py).
        eta_oracle = energy(a, z - pref * r)
        dq = fro(q.T * q - mp.eye(q.cols))  # Safe Frobenius upper bound for 2-norm.
        qbar = mp.sqrt(1 + dq)
        hlow = ell_a * (1 - dq)
        assert hlow > 0
        dcbar = qbar * fro(ew) + fro(ec)
        abar = (fro(ka) + fro(ed) + dcbar * fro(alpha)) / mp.sqrt(hlow)
        if u is None:
            ell = energy(a, q * t - z)
            rbarprev = cbarprev = bbar = mp.mpf(0)
            current_eqr = q * t - z
        else:
            current_eqr = q * t - vold
            prev = previous_defects
            ell = energy(a, prev["ev"]) + energy(a, current_eqr)
            rbarprev = (
                fro(prev["ka"])
                + fro(prev["ed"])
                + fro(prev["ec"]) * fro(prev["alpha"])
                + prev["qbar"] * fro(prev["er"])
            )
            cbarprev = (
                fro(prev["kb"])
                + fro(prev["ef"])
                + fro(prev["ew"]) * norm2(z)
                + prev["dcbar"] * fro(bold)
                + mp.sqrt(a_plus) * prev["qbar"] * ell
            )
            bbar = (rbarprev + cbarprev) / mp.sqrt(prev["hlow"])
        unext = mp.sqrt(
            (chi * ubound + eta_oracle + (1 + chi) * gbar + ell) ** 2
            + (abar + gbar) ** 2
            + (0 if u is None else (bbar + gbar) ** 2)
        ) + energy(a, ex)
        gnextbar = (
            gbar
            + fro(ew) * norm2(alpha) / mp.sqrt(ell_a)
            + fro(er) / mp.sqrt(ell_a)
            + energy(a, ex)
        )
        budget_slacks = {
            "current_error": ubound - energy(a, e),
            "current_gap": gbar - gapnorm,
            "current_alpha_projection": abar - invenergy(h, da),
            "current_QR_update": ell - energy(a, l),
            "next_error": unext - actual_next,
            "next_gap": gnextbar - invenergy(a, gn),
        }
        if u is not None:
            budget_slacks.update(
                previous_residual=rbarprev - fro(u.T * r),
                previous_weighted_conjugacy=cbarprev - fro(cw),
                previous_projection=bbar - invenergy(u.T * a * u, previous_projection_residual),
            )
        # An exact A projection based on the true algebraic residual.
        alpha_sharp = solve(h, q.T * exact_r)
        xsharp = x + q * alpha_sharp
        gaprhs = g - ew * alpha + er + a * ex
        local_rhs = ka + ec * alpha - ed + q.T * er
        conjugacy_rhs = -kb - ef - ew.T * zn - (ec + q.T * ew) * beta + q.T * a * ev
        conjugacy_lhs = q.T * a * v
        qr_projection_lhs = q.T * a * qn * tn
        qr_projection_rhs = conjugacy_lhs + q.T * a * eqr
        projection_pythag_lhs = energy(a, e) ** 2
        projection_pythag_rhs = energy(a, xstar - xsharp) ** 2 + energy(a, xsharp - x) ** 2
        identities = {
            "gap_recurrence": scaled_defect(
                gn, gaprhs, fro(g) + fro(ew * alpha) + fro(er) + fro(a * ex) + fro(gn)
            ),
            "local_residual": scaled_defect(
                q.T * rn,
                local_rhs,
                fro(q.T * rn) + fro(ka) + fro(ec * alpha) + fro(ed) + fro(q.T * er),
            ),
            "local_conjugacy_before_QR": scaled_defect(
                conjugacy_lhs,
                conjugacy_rhs,
                fro(conjugacy_lhs)
                + fro(kb)
                + fro(ef)
                + fro(ew.T * zn)
                + fro((ec + q.T * ew) * beta)
                + fro(q.T * a * ev),
            ),
            "QR_reconstruction": scaled_defect(qn * tn, zn + q * beta + ev + eqr),
            "QR_weighted_conjugacy": scaled_defect(
                qr_projection_lhs,
                qr_projection_rhs,
                fro(qr_projection_lhs) + fro(conjugacy_lhs) + fro(q.T * a * eqr),
            ),
            "current_L_decomposition": scaled_defect(
                l, current_l_parts, fro(q * t) + fro(z) + (0 if u is None else fro(u * bold))
            ),
            "true_projection_orthogonality": fro(q.T * (b - a * xsharp))
            / max(fro(q) * fro(exact_r), mp.mpf("1e-200")),
            "true_projection_Pythagoras": scaled_defect(
                projection_pythag_lhs, projection_pythag_rhs
            ),
            "new_one_step_identity": scaled_defect(theorem_lhs, theorem_rhs, theorem_scale),
        }
        diagnostic = {
            "j": row["j"],
            "identity_scaled_defects": {k: number(v) for k, v in identities.items()},
            "true_A_error_relative_to_initial_A_error": number(energy(a, e) / initial_error),
            "next_true_A_error_relative_to_initial_A_error": number(actual_next / initial_error),
            "true_residual_F": number(fro(exact_r)),
            "recursive_residual_F": number(fro(r)),
            "true_gap_Ainv_F": number(gapnorm),
            "actual_inner_relative_error_Theta": number(theta),
            "Q_orthogonality_F": number(fro(q.T * q - mp.eye(q.cols))),
            "Q_full_column_rank_exact_rational_check": True,
            "current_L_A_F": number(energy(a, l)),
            "X_update_defect_A_F": number(energy(a, ex)),
            "dalpha_Hinv_F": number(invenergy(h, da)),
            "one_step_identity_lhs": number(theorem_lhs),
            "one_step_identity_rhs": number(theorem_rhs),
            "completed_square_bound_squared": number(completed_square),
            "completed_square_slack": number(completed_square - theorem_lhs),
            "stored_next_bound": number(stored_bound),
            "actual_stored_next_error": number(actual_next),
            "stored_bound_slack": number(stored_bound - actual_next),
            "Theta_substituted_stored_bound": number(theta_bound),
            "Theta_substituted_slack": number(theta_bound - actual_next),
            "stored_bound_over_current_error": number(
                stored_bound / max(energy(a, e), mp.mpf("1e-200"))
            ),
            "actual_next_over_current_error": number(
                actual_next / max(energy(a, e), mp.mpf("1e-200"))
            ),
            "current_rank_estimate_fp64": row["qr_current"]["rank_estimate_fp64"],
            "current_block_width": q.cols,
            "recursive_oracle_budget_check": {
                "eta_input_kind": "measured ||Z-P R||_A,F",
                "chi_analytic_upper": number(chi),
                "eta_oracle": number(eta_oracle),
                "U_current": number(ubound),
                "gbar_current": number(gbar),
                "ell_current": number(ell),
                "abar_current": number(abar),
                "rbar_previous": number(rbarprev),
                "cweightedbar_previous": number(cbarprev),
                "bbar_previous": number(bbar),
                "deltaQ_F_upper": number(dq),
                "qbar_current": number(qbar),
                "H_lower_current": number(hlow),
                "deltaCbar_current": number(dcbar),
                "U_next": number(unext),
                "gbar_next": number(gnextbar),
                "slacks": {key: number(val) for key, val in budget_slacks.items()},
            },
        }
        if previous_projection_residual is not None:
            diagnostic["previous_projection_residual_HUinv_F"] = number(
                invenergy(u.T * a * u, previous_projection_residual)
            )
        results.append(diagnostic)
        # gbar is advanced only after its use above.
        gbar, ubound = gnextbar, unext
        previous_defects = dict(
            ka=ka,
            kb=kb,
            ew=ew,
            ec=ec,
            ed=ed,
            er=er,
            ef=ef,
            ev=ev,
            alpha=alpha,
            qbar=qbar,
            hlow=hlow,
            dcbar=dcbar,
        )
    max_identity = max(
        mp.mpf(value) for row in results for value in row["identity_scaled_defects"].values()
    )
    slacks = [
        mp.mpf(row[field])
        for row in results
        for field in ("completed_square_slack", "stored_bound_slack", "Theta_substituted_slack")
    ]
    # Small allowances cover mpmath rounding in this check (no directed rounding here).
    bad_bound = [
        row["j"]
        for row in results
        if mp.mpf(row["stored_bound_slack"]) < -mp.mpf("1e-85") * initial_error
        or mp.mpf(row["Theta_substituted_slack"]) < -mp.mpf("1e-85") * initial_error
        or mp.mpf(row["completed_square_slack"]) < -mp.mpf("1e-85") * initial_error**2
    ]
    assert max_identity < mp.mpf("1e-70"), (record["spec"]["name"], max_identity)
    assert not bad_bound, (record["spec"]["name"], bad_bound)
    recursive_bad = [
        {"j": row["j"], "budget": key, "slack": value}
        for row in results
        for key, value in row["recursive_oracle_budget_check"]["slacks"].items()
        if mp.mpf(value) < -mp.mpf("1e-80") * initial_error
    ]
    assert not recursive_bad, (record["spec"]["name"], recursive_bad)
    return {
        "name": record["spec"]["name"],
        "reference_relative_residual_at_100_dps": number(reference_defect),
        "reference_relative_change_100_vs_160_dps": number(reference_comparison),
        "max_identity_scaled_defect": number(max_identity),
        "bound_violations": bad_bound,
        "recursive_oracle_budget_violations": recursive_bad,
        "eta_budget_status": "measured from the stored trajectory",
        "chi_analytic_upper_exact_rational": chi_rational,
        "final_recursive_oracle_U_over_initial_A_error": number(ubound / initial_error),
        "rank_deficient_QR_inputs": sum(
            r["current_rank_estimate_fp64"] < r["current_block_width"] for r in results
        ),
        "final_true_A_error_relative_to_initial_A_error": results[-1][
            "next_true_A_error_relative_to_initial_A_error"
        ],
        "inner_underflow_to_zero_count": sum(
            row["bjac_current"]["underflow_to_zero"] for row in record["steps"]
        ),
        "inner_unchanged_sweeps_count": sum(
            row["bjac_current"]["unchanged_sweeps"] for row in record["steps"]
        ),
        "inner_unchanged_sweeps_with_nonzero_true_residual_count": stagnation_with_nonzero_true_inner_residual,
        "inner_unchanged_sweeps_with_nonzero_recursive_residual_count": sum(
            row["bjac_current"]["stagnated_sweeps_with_nonzero_recursive_inner_residual"]
            for row in record["steps"]
        ),
        "steps": results,
    }


def jsonable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [jsonable(v) for v in obj]
    return obj


def load_record(path):
    record = json.loads(path.read_text())
    for key in ("A", "B", "X0"):
        record["spec"][key] = np.asarray(record["spec"][key], dtype=np.float64)
    for key in ("X", "R", "Z", "Q", "T"):
        record["initial"][key] = np.asarray(record["initial"][key], dtype=np.float64)
    for row in record["steps"]:
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
            "Qnext",
            "Tnext",
            "previous_Q",
            "previous_beta",
            "previous_V",
        ):
            if row[key] is not None:
                row[key] = np.asarray(row[key], dtype=np.float64)
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=HERE / "results")
    parser.add_argument("--case", default=None)
    parser.add_argument(
        "--replay",
        type=Path,
        default=None,
        help="re-check stored traces in this directory instead of recomputing them",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    specs = [
        make_case("random_s3_fp32", 3, "fp32"),
        make_case("random_s3_fp16", 3, "fp16"),
        make_case("single_rhs_fp32", 1, "fp32"),
        make_case("nonzero_initial_s3_fp32", 3, "fp32", nonzero_x=True),
        make_case("near_correlated_s3_fp32", 3, "fp32", rhs_mode="near"),
        make_case("near_correlated_s3_fp16", 3, "fp16", rhs_mode="near"),
        make_case("exact_correlated_s3_fp32", 3, "fp32", rhs_mode="exact"),
        make_case("underflow_s3_fp16", 3, "fp16", scale=1e-10),
        make_case("inner_stagnation_s3_fp16", 3, "fp16", k=8, t=4),
    ]
    if args.case:
        specs = [s for s in specs if s["name"] == args.case]
        if not specs:
            raise ValueError("Unknown --case")
    summary = {
        "precision_digits": mp.mp.dps,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "mpmath": mp.__version__,
        "platform": platform.platform(),
        "algorithm": "two-level columnwise rounded BJAC plus raw-QR variable Block CG",
        "note": "High-precision checks; rigorous bounds are produced by certify_budgets.py.",
        "normalization": "True A-norm errors are divided by ||A^-1 B-X0||_A,F.",
        "recursive_budget_scope": "Local defects and eta=||Z-P R||_A,F are measured from the trajectory.",
        "cases": [],
    }
    for spec in specs:
        record = (
            run_case(spec)
            if args.replay is None
            else load_record(args.replay / (spec["name"] + "_trace.json"))
        )
        audit = audit_case(record)
        rawpath = args.output / (spec["name"] + "_trace.json")
        rawpath.write_text(
            json.dumps(jsonable(record), ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        )
        auditpath = args.output / (spec["name"] + "_audit.json")
        auditpath.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
        case_summary = {k: v for k, v in audit.items() if k != "steps"}
        case_summary["trace_sha256"] = hashlib.sha256(rawpath.read_bytes()).hexdigest()
        case_summary["audit_sha256"] = hashlib.sha256(auditpath.read_bytes()).hexdigest()
        summary["cases"].append(case_summary)
        print(
            spec["name"],
            "max identity defect",
            audit["max_identity_scaled_defect"],
            "QR deficient",
            audit["rank_deficient_QR_inputs"],
            flush=True,
        )
    summary["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (args.output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    report = [
        "# Summary of high-precision checks (100 digits)",
        "",
        "| case | final error / initial error | U / initial error | rank-deficient QR inputs | max identity defect |",
        "|---|---:|---:|---:|---:|",
    ]
    for c in summary["cases"]:
        report.append(
            "| {} | {:.6e} | {:.6e} | {} | {:.3e} |".format(
                c["name"],
                float(c["final_true_A_error_relative_to_initial_A_error"]),
                float(c["final_recursive_oracle_U_over_initial_A_error"]),
                c["rank_deficient_QR_inputs"],
                float(c["max_identity_scaled_defect"]),
            )
        )
    report += [
        "",
        "Errors are Frobenius energy norms after 18 updates. Rank deficiency is an FP64 SVD diagnostic.",
    ]
    (args.output / "SUMMARY.md").write_text("\n".join(report) + "\n")


if __name__ == "__main__":
    main()
