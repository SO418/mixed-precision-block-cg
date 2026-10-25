"""QR-based block CG with the recurrence bound, and two reference variants."""

import math, time
import numpy as np
from numerics import dot_env, add_env, sparse_env


class IdentityPreconditioner:
    # Scalar scaling changes no exact CG search space. It supplies a contracting P.
    def __init__(self, m):
        self.m = m
        self.scale = 2 / (m.amin + m.aplus)
        self.chi = (m.aplus - m.amin) / (m.aplus + m.amin)
        self.nu = 2 * m.aplus / (m.amin + m.aplus)

    def __call__(self, r):
        z = self.scale * r
        eta = math.sqrt(self.m.aplus) * self.m.be.norm(2.0**-53 * abs(z) + 2.0**-1074)
        return z, dict(
            eta=eta,
            coarse_eta=eta,
            cast_part=0.0,
            rounding_part=eta,
            input_zero_loss_count=0,
            condition_ok=self.chi < 1,
            finite=True,
            strict_certificate=False,
        )


def solve(m, b, xref, pre, spec, refmeta, reference_pre=None):
    xp = m.xp
    be = m.be
    norm = be.norm
    rtlo = math.sqrt(m.amin)
    rthi = math.sqrt(m.aplus)
    x = xp.zeros_like(b)
    r = b.copy()
    bn = xp.linalg.norm(b, axis=0)
    chi = getattr(pre, "chi", m.chi)
    nu = getattr(pre, "nu", m.nu)
    U = norm(b) / rtlo
    gamma = 0.0
    prev = None
    vbound = 0.0
    hist = []
    calls = []
    xnorm = norm(xref)
    normscale = rthi * xnorm
    referr = refmeta["reference_residual_norm_upper"] / rtlo
    e0 = math.sqrt(max(0.0, float(xp.sum(xref * m.mv(xref)))))
    maxit = spec.get("maxit", 600)
    patience = spec.get("patience", 60)
    minit = spec.get("minit", 80)
    best = float("inf")
    last_improve = 0
    status = "maxit"
    failure = None
    first_target = None
    bound_prefix = True
    be.sync()
    start = time.perf_counter()

    def observe(k, bud=None):
        true = b - m.mv(x)
        diff = xref - x
        en = math.sqrt(max(0.0, float(xp.sum(diff * m.mv(diff)))))
        # Reference error from exact dyadic RHS residual is separated from the measured curve.
        rr = xp.linalg.norm(true, axis=0) / bn
        rr_env = xp.linalg.norm(sparse_env(m, x) + add_env(b, -m.mv(x), xp), axis=0) / bn
        row = {
            "iteration": k,
            "forward_A_reference": en,
            "forward_A_lower": max(0.0, en - referr),
            "forward_A_upper": en + referr,
            "forward_scaled": en / normscale,
            "true_relative_residual_max": float(xp.max(rr)),
            "true_relative_residual_lower_est": max(0.0, float(xp.max(rr - rr_env))),
            "true_relative_residual_upper_est": float(xp.max(rr + rr_env)),
            "true_relative_residual_columns": be.host(rr).tolist(),
            "scaled_true_residual": norm(true) / (m.aplus * xnorm),
            "normwise_backward_error_est": norm(true) / (m.aplus * norm(x) + norm(b)),
            "recursive_relative_residual_max": float(xp.max(xp.linalg.norm(r, axis=0) / bn)),
            "gap_norm": norm(r - true),
            "error_over_initial": en / e0,
        }
        if bud:
            row.update(bud)
        hist.append(row)
        return row

    def direction(v):
        if spec.get("algorithm", "qr") == "raw":
            return v.copy(), xp.eye(v.shape[1], dtype=xp.float64)
        return xp.linalg.qr(v, mode="reduced")

    def call(r, k):
        z, info = pre(r)
        info = dict(info, call=k)
        info.pop("last_block_L", None)
        if reference_pre is not None and k % spec.get("sample_every", 10) == 0 and info["finite"]:
            zr, ib = reference_pre(r)
            diff = z - zr
            distance = math.sqrt(max(0.0, float(xp.sum(diff * m.mv(diff)))))
            info.update(
                reference_difference_A=distance,
                application_error_lower_est=max(0.0, distance - ib["eta"]),
                application_error_upper_est=distance + ib["eta"],
                reference_eta=ib["eta"],
                application_bound_numerical_check=distance <= info["eta"] + ib["eta"],
            )
        # The same input-dependent call target is used for every configuration.
        target = spec.get("call_rtol", 1e-3) * norm(r) / rtlo
        info["call_target"] = target
        info["call_target_met"] = info["eta"] <= target
        calls.append(info)
        if not info["finite"]:
            raise FloatingPointError("nonfinite_preconditioner")
        # A sum-of-squares norm may underflow even when entries are nonzero.
        if bool(xp.all(z == 0)) and bool(xp.any(r != 0)):
            raise FloatingPointError("zero_preconditioner_output")
        return z, info

    observe(0)
    try:
        z, bi = call(r, 0)
        q, T = direction(z)
        qrinput = z
        for k in range(maxit):
            gram = q.T @ q
            orth_env = (
                abs(gram - xp.eye(q.shape[1]))
                + dot_env(q.T, q, xp)
                + add_env(gram, -xp.eye(q.shape[1]), xp)
            )
            delta = norm(orth_env)
            # Normalized QR uses the formula from the paper; raw directions use spectral estimates.
            if delta < 1:
                h = m.amin * (1 - delta)
                qn = math.sqrt(1 + delta)
            else:
                eigen = be.host(xp.linalg.eigvalsh((gram + gram.T) / 2))
                gram_error = norm(dot_env(q.T, q, xp))
                h = m.amin * (float(eigen[0]) - gram_error)
                qn = math.sqrt(max(0.0, float(eigen[-1]) + gram_error))
            rank_ok = h > 0 and math.isfinite(h)
            # Failure of a sufficient rank/bound check does not stop the solver.
            # Keep the unmodified method running and mark its bound unavailable.
            w = m.mv(q)
            c = q.T @ w
            d = q.T @ r
            alpha = xp.linalg.solve(c, d)
            if not bool(xp.all(xp.isfinite(alpha))):
                raise FloatingPointError("nonfinite_small_solve")
            qa = q @ alpha
            wa = w @ alpha
            xn = x + qa
            rn = r - wa
            wg = norm(sparse_env(m, q))
            cg = norm(dot_env(q.T, w, xp))
            dg = norm(dot_env(q.T, r, xp))
            xg = rthi * norm(dot_env(q, alpha, xp) + add_env(x, qa, xp))
            rg = norm(dot_env(w, alpha, xp) + add_env(r, -wa, xp))
            ca = c @ alpha
            sa = norm(abs(d - ca) + dot_env(c, alpha, xp) + add_env(d, -ca, xp))
            qt = q @ T
            qrg = rthi * norm(abs(qt - qrinput) + dot_env(q, T, xp) + add_env(qt, -qrinput, xp))
            ell = qrg + vbound
            epsc = qn * wg + cg
            aa = (sa + dg + epsc * norm(alpha)) / math.sqrt(h) if rank_ok else float("nan")
            bb = 0.0
            if prev is not None:
                jj = (
                    prev["sb"]
                    + prev["f"]
                    + prev["w"] * norm(z)
                    + prev["epsc"] * norm(prev["beta"])
                    + prev["qA"] * ell
                )
                bb = (prev["tbar"] + jj) / math.sqrt(prev["h"]) if prev["h"] > 0 else float("nan")
            tbar = sa + dg + cg * norm(alpha) + qn * rg
            zeta = bi["eta"] + nu * gamma + ell
            omega = (aa + gamma) ** 2 + ((bb + gamma) ** 2 if prev is not None else 0.0)
            condition = bool(
                rank_ok and bi["condition_ok"] and chi < 1 and epsc < h and math.isfinite(bb)
            )
            bound_prefix = bound_prefix and condition
            Unext = (
                math.hypot(chi * U + zeta, math.sqrt(omega)) + xg if bound_prefix else float("nan")
            )
            gnext = gamma + (wg * norm(alpha) + rg) / rtlo + xg
            bud = {
                "eta": bi["eta"],
                "coarse_eta": bi["coarse_eta"],
                "ell": ell,
                "a_budget": aa,
                "b_budget": bb,
                "gamma": gamma,
                "gamma_next": gnext,
                "zeta": zeta,
                "sqrt_omega": math.sqrt(omega),
                "x_budget": xg,
                "bound_A": Unext,
                "bound_scaled": Unext / normscale,
                "gap_baseline_A": norm(rn) / rtlo + gnext,
                "true_residual_F_bound": rthi * Unext,
                "scaled_true_residual_bound": Unext / normscale,
                "relative_residual_max_bound": rthi * Unext / float(xp.min(bn)),
                "chi": chi,
                "nu": nu,
                "h": h,
                "eps_C": epsc,
                "orthogonality_delta_est": delta,
                "direction_rank_bound_available": rank_ok,
                "bound_conditions_this_step": condition,
                "bound_prefix_available": bound_prefix,
                "strict_certificate": False,
            }
            x, r = xn, rn
            row = observe(k + 1, bud)
            row["bound_numerical_check"] = (
                (row["forward_A_upper"] <= Unext) if bound_prefix else None
            )
            row["bound_over_forward_error"] = (
                Unext / row["forward_A_reference"]
                if bound_prefix and row["forward_A_reference"] > 0
                else None
            )
            row["bound_over_forward_upper"] = (
                Unext / row["forward_A_upper"]
                if bound_prefix and row["forward_A_upper"] > 0
                else None
            )
            row["residual_bound_numerical_check"] = (
                (row["true_relative_residual_lower_est"] <= bud["relative_residual_max_bound"])
                if bound_prefix
                else None
            )
            if first_target is None and row["true_relative_residual_upper_est"] <= spec.get(
                "tol", 1e-12
            ):
                first_target = k + 1
            objective = row["forward_A_upper"]
            if not math.isfinite(objective):
                status = "nonfinite"
                break
            if objective < 0.95 * best:
                best = objective
                last_improve = k + 1
            if (k + 1) >= minit and (k + 1 - last_improve) >= patience:
                status = "plateau_observed"
                break
            if k + 1 == maxit:
                break
            zn, bninfo = call(r, k + 1)
            if spec.get("algorithm", "qr") == "restart":
                v = zn
                prev = None
                vbound = 0.0
            else:
                f = w.T @ zn
                beta = -xp.linalg.solve(c, f)
                qb = q @ beta
                v = zn + qb
                cb = c @ beta
                prev = {
                    "w": wg,
                    "f": norm(dot_env(w.T, zn, xp)),
                    "sb": norm(abs(-f - cb) + dot_env(c, beta, xp) + add_env(-f, -cb, xp)),
                    "epsc": epsc,
                    "beta": beta,
                    "h": h,
                    "qA": rthi * qn,
                    "tbar": tbar,
                }
                vbound = rthi * norm(dot_env(q, beta, xp) + add_env(zn, qb, xp))
            q, T = direction(v)
            qrinput = v
            z, bi = zn, bninfo
            U, gamma = Unext, gnext
    except (FloatingPointError, np.linalg.LinAlgError) as exc:
        status = "numerical_breakdown"
        failure = str(exc)
    be.sync()
    elapsed = time.perf_counter() - start
    bestrow = min(hist, key=lambda row: row["forward_A_upper"])
    return {
        "status": status,
        "error": failure,
        "history": hist,
        "calls": calls,
        "updates": len(hist) - 1,
        "first_target_iteration": first_target,
        "best_forward_A_upper": bestrow["forward_A_upper"],
        "best_forward_iteration": bestrow["iteration"],
        "residual_at_best_forward": bestrow["true_relative_residual_max"],
        "scaled_residual_at_best_forward": bestrow["scaled_true_residual"],
        "backward_proxy_at_best_forward": bestrow["normwise_backward_error_est"],
        "best_true_relative_residual": min(r["true_relative_residual_max"] for r in hist),
        "final_true_relative_residual": hist[-1]["true_relative_residual_max"],
        "all_calls_meet_target": all(r["call_target_met"] for r in calls),
        "numerical_bound_violations": sum(r.get("bound_numerical_check") is False for r in hist),
        "residual_bound_violations": sum(
            r.get("residual_bound_numerical_check") is False for r in hist
        ),
        "application_bound_violations": sum(
            r.get("application_bound_numerical_check") is False for r in calls
        ),
        "diagnostic_wall_seconds": elapsed,
        "timing_is_performance_measurement": False,
        "reference_error_A_upper": referr,
        "normalization_sqrt_Aupper_times_Xref": normscale,
        "strict_certificate": False,
    }
