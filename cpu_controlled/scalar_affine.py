"""Conventional scalar PCG with its error bounds, and an input-independent affine BJAC bound.

Order-16 test matrix, seeds 301-303, four inner formats. Bounds use stored operands and
rational arithmetic; the exact solution is used only for validation.

Usage: python3 scalar_affine.py --out results
"""

from pathlib import Path
import argparse, sys, json, hashlib, platform
from fractions import Fraction as F
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "exact"))
from verify_qr_bjac import Arithmetic, bjac, jsonable
from certify_budgets import (
    sqrt_interval,
    fm,
    eye,
    ss,
    root,
    up,
    energy2,
    solve_exact,
    family_constants,
    bjac_budget,
    reference_p,
    serial,
)
from formats import conversion_model

OUT = Path("results")


def save(name, v):
    (OUT / name).write_text(json.dumps(serial(v), indent=2))


def dot(a, b):
    return (a.T @ b)[0, 0]


def problem(seed):
    a = np.eye(16)
    for i in range(15):
        a[i, i + 1] = a[i + 1, i] = -float(F(1, 64) if (i + 1) % 4 == 0 else F(3, 8))
    b = np.random.default_rng(seed).standard_normal((16, 1))
    return a, b


def scalar(seed, precision):
    a, b = problem(seed)
    af, bf = fm(a), fm(b)
    sp = dict(blocksize=4, k=2, t=3, precision=precision)
    co = family_constants(af, 4, 2, 3)
    chi = co["chi"]
    nu = 1 + chi
    p = reference_p(af, sp)
    xs = solve_exact(af, bf)
    e02 = energy2(af, xs)
    op = Arithmetic("fp64")
    x = np.zeros_like(b)
    r = op.q(b - op.mm(a, x))
    z, stats = bjac(a, r, 4, 2, 3, precision)
    d = z.copy()
    rho = op.mm(r.T, z)[0, 0]
    prev = None
    beta = None
    U = root(ss(bf) / co["amin"])
    rows = []
    traces = []
    primitive = 0
    status = "maxit"
    for k in range(20):
        if not rho or not np.any(d):
            status = "zero_direction_or_rho"
            break
        v = op.mm(a, d)
        sigma = op.mm(d.T, v)[0, 0]
        if not sigma:
            status = "zero_denominator"
            break
        theta = float(op.q(rho / sigma))
        xn = op.q(x + op.q(theta * d))
        rn = op.q(r - op.q(theta * v))
        xf, rf, zf, df, vf, xnf, rnf = map(fm, [x, r, z, d, v, xn, rn])
        tf = F(theta)
        if prev is None:
            ed = df - zf
        else:
            ed = df - zf - F(beta) * fm(prev)
        ex = xnf - xf - tf * df
        er = rnf - rf + tf * af @ df
        g = rf - (bf - af @ xf)
        h = energy2(af, df)
        ab = root((dot(df, rf) - tf * h) ** 2 / h)
        bb = (
            F(0)
            if prev is None
            else root((dot(fm(prev), rf) - dot(fm(prev), af @ df)) ** 2 / energy2(af, fm(prev)))
        )
        ell = root(energy2(af, ed))
        xb = root(energy2(af, ex))
        gamma = root(ss(g) / co["amin"])
        bj = bjac_budget(a, r, z, stats, sp, co)
        eta = bj["eta"]
        primitive += bj["primitive_checks"]
        omega = (ab + gamma) ** 2 + (F(0) if prev is None else (bb + gamma) ** 2)
        Un = up(root((chi * U + eta + nu * gamma + ell) ** 2 + omega) + xb)
        e2 = energy2(af, xs - xnf)
        eta2 = energy2(af, zf - p @ rf)
        assert e2 <= Un * Un and eta2 <= eta * eta
        gn = rnf - (bf - af @ xnf)
        assert np.array_equal(gn, g + er + af @ ex)
        # Exact identity rebuilt from the stored scalar coefficients.
        E = xs - xf
        K = 1 - tf
        da = dot(df, rf) - tf * h
        lhs = energy2(af, E - tf * df)
        rhs = energy2(af, E - (zf + ed)) - K * K * h + 2 * K * (da - dot(df, g))
        if prev is not None:
            pf = fm(prev)
            betaf = F(beta)
            J = dot(pf, af @ df)
            rhs -= betaf * betaf * energy2(af, pf) + 2 * betaf * (dot(pf, rf) - J - dot(pf, g))
        assert lhs == rhs
        base = up(root(ss(rnf) / co["amin"]) + root(ss(gn) / co["amin"]))
        assert e2 <= base * base
        res2 = ss(bf - af @ xnf)
        rows.append(
            dict(
                update=k + 1,
                error=float(root(e2 / e02)),
                bound=float(root(Un * Un / e02)),
                baseline=float(root(base * base / e02)),
                a=float(ab),
                b=float(bb),
                eta=float(eta),
                gap=float(gamma),
                ell=float(ell),
                x=float(xb),
                true_residual=float(root(res2 / ss(bf))),
                bound_over_error=float(root(Un * Un / e2)),
            )
        )
        traces.append(
            dict(
                X=x,
                R=r,
                Z=z,
                d=d,
                theta=theta,
                Xnext=xn,
                Rnext=rn,
                previous_d=prev,
                beta=beta,
                bjac=stats,
                budget=dict(U=U, Unext=Un, a=ab, b=bb, gamma=gamma, ell=ell, x=xb, eta=eta),
                actual_error_squared=e2,
            )
        )
        x, r, U = xn, rn, Un
        if res2 <= F(1, 10**24) * ss(bf):
            status = "target_met"
            break
        zn, sn = bjac(a, r, 4, 2, 3, precision)
        rhon = op.mm(r.T, zn)[0, 0]
        beta = float(op.q(rhon / rho))
        prev = d.copy()
        d = op.q(zn + op.q(beta * d))
        z, stats, rho = zn, sn, rhon
    name = f"scalar_{precision}_{seed}"
    save(name + "_trace.json", jsonable(dict(A=a, B=b, spec=sp, steps=traces)))
    return dict(
        id=name,
        precision=precision,
        seed=seed,
        status=status,
        steps=len(rows),
        chi=float(chi),
        primitive_checks=primitive,
        exact_energy_identities=len(rows),
        exact_forward_comparisons=len(rows),
        rows=rows,
    )


# Pairs (relative coefficient, absolute coefficient) majorize each scalar magnitude
# as c*||r||_{A^-1}+d. All operations below preserve nonnegative affine bounds.
def pair_add(a, b):
    return tuple(up(x + y) for x, y in zip(a, b))


def pair_scale(c, a):
    return tuple(up(c * x) for x in a)


def affine_budget(a, co, precision):
    ul, tl = conversion_model(precision)
    uh, th = F(1, 2**53), F(1, 2**1075)
    u = uh + ul + uh * ul
    tau = (1 + ul) * th + tl
    af = abs(fm(a))
    n = len(a)
    zero = (F(0), F(0))

    def rnd(x):
        return pair_add(pair_scale(1 + u, x), (F(0), tau))

    def err(x):
        return pair_add(pair_scale(u, x), (F(0), tau))

    def norm(v):
        return (root(sum((x[0] ** 2 for x in v), F(0))), root(sum((x[1] ** 2 for x in v), F(0))))

    def mm(mat, v):
        vals = []
        errors = []
        for row in mat:
            accum = zero
            eb = zero
            for c, xx in zip(row, v):
                prod = pair_scale(c, xx)
                stored = rnd(prod)
                sumv = pair_add(accum, stored)
                eb = pair_add(eb, pair_add(err(prod), err(sumv)))
                accum = rnd(sumv)
            vals.append(accum)
            errors.append(eb)
        return vals, errors

    def sub(v, w):
        sums = [pair_add(x, y) for x, y in zip(v, w)]
        return [rnd(x) for x in sums], [err(x) for x in sums]

    # |r_i| <= sqrt(aplus) ||r||_{A^-1}. Input coefficients are exactly stored.
    inp = [(root(co["aplus"]), F(0))] * n
    rhs = [rnd(x) for x in inp]
    cast = [err(x) for x in inp]
    castb = pair_scale((1 + co["chi"]) * root(1 / co["amin"]), norm(cast))
    y = [zero] * n
    prop = zero
    for sw in range(2):
        ay, ew = mm(af, y)
        res, er = sub(rhs, ay)
        eres = [pair_add(x, z) for x, z in zip(ew, er)]
        correction = []
        lis = []
        for block, beg in enumerate(range(0, n, 4)):
            mat = af[beg : beg + 4, beg : beg + 4]
            loc = [zero] * 4
            lb = zero
            qb = co["block_q"][block]
            for _ in range(3):
                w, wg = mm(mat, loc)
                rr, rv = sub(res[beg : beg + 4], w)
                # Division by exactly one still receives the generic rounding enclosure.
                du = [rnd(x) for x in rr]
                rc = [err(x) for x in rr]
                un, ra = sub(loc, du)
                defects = [
                    pair_add(pair_add(a, b), pair_add(c, d)) for a, b, c, d in zip(wg, rv, rc, ra)
                ]
                lb = pair_add(pair_scale(qb, lb), norm(defects))
                # Bound the new state through u_new=(I-Aii)u+s+defect,
                # retaining cancellation of the unit diagonal in the exact recurrence.
                jm = mat.copy()
                np.fill_diagonal(jm, F(0))
                loc = [
                    pair_add(
                        pair_add(
                            tuple(sum((c * v[h] for c, v in zip(row, loc)), F(0)) for h in [0, 1]),
                            rs,
                        ),
                        de,
                    )
                    for row, rs, de in zip(jm, res[beg : beg + 4], defects)
                ]
            smin = (1 - qb**3) / (1 + qb)
            lis.append(pair_scale(root(1 / smin), lb))
            correction.extend(loc)
        yn, eupd = sub(y, correction)
        cr = pair_scale(root(co["betab"] * co["bt"]), norm(eres))
        ci = pair_scale(root(co["betab"]), norm(lis))
        cu = pair_scale(root(co["aplus"]), norm(eupd))
        prop = pair_add(pair_scale(co["rho"], prop), pair_add(pair_add(cr, ci), cu))
        y = yn
    return pair_add(castb, prop)


def affine_tests():
    out = []
    for precision in ["fp64", "fp32", "fp16", "bf16"]:
        a, _ = problem(301)
        af = fm(a)
        sp = dict(blocksize=4, k=2, t=3, precision=precision)
        co = family_constants(af, 4, 2, 3)
        p = reference_p(af, sp)
        eps, delta = affine_budget(a, co, precision)
        rows = []
        for seed in [301, 302, 303]:
            _, b = problem(seed)
            for exponent in [0, -20, -40]:
                r = np.ldexp(b, exponent)
                z, stats = bjac(a, r, 4, 2, 3, precision)
                rf, zf = fm(r), fm(z)
                nr2 = dot(rf, solve_exact(af, rf))
                eta2 = energy2(af, zf - p @ rf)
                # Lower enclosure of the square root for the exact inequality test.
                nrlo, nrhi = sqrt_interval(nr2)
                boundlo = eps * nrlo + delta
                assert eta2 <= boundlo * boundlo
                rows.append(
                    dict(
                        seed=seed,
                        scale_exponent=exponent,
                        input_norm=float(nrhi),
                        actual_eta=float(root(eta2)),
                        relative_part=float(eps * nrhi),
                        absolute_part=float(delta),
                        bound=float(eps * nrhi + delta),
                        underflows=stats["underflow_to_zero"],
                    )
                )
        out.append(
            dict(
                precision=precision,
                epsilon=float(eps),
                delta=float(delta),
                epsilon_exact=str(eps),
                delta_exact=str(delta),
                chi=float(co["chi"]),
                chi_plus_epsilon=float(co["chi"] + eps),
                rows=rows,
            )
        )
    return out


if __name__ == "__main__":
    pa = argparse.ArgumentParser()
    pa.add_argument("--out", type=Path, default=Path("results"))
    OUT = pa.parse_args().out
    OUT.mkdir(parents=True, exist_ok=True)
    scal = []
    for precision in ["fp64", "fp32", "fp16", "bf16"]:
        for seed in [301, 302, 303]:
            q = scalar(seed, precision)
            scal.append(q)
            save("scalar_summary.json", scal)
            print(q["id"], q["status"], q["steps"], flush=True)
    aff = affine_tests()
    save("affine_summary.json", aff)
    save(
        "environment.json",
        dict(
            python=sys.version,
            numpy=np.__version__,
            platform=platform.platform(),
            source_hashes={
                p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in [HERE / "scalar_affine.py", *(HERE.parent / "exact").glob("*.py")]
            },
        ),
    )
    print("affine", [(q["precision"], q["epsilon"], q["delta"]) for q in aff], flush=True)
