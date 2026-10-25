"""Test problems built from integers so that B = A X* holds exactly in FP64."""

import numpy as np


def problem(spec):
    g = spec["g"]
    n = g**3
    s = spec["rhs"]
    family = spec["family"]
    scale = spec.get("rhs_exp", 0)
    ce = spec.get("corr_exp")
    xyz = np.stack(np.unravel_index(np.arange(n), (g, g, g)), axis=1)
    ids = np.arange(n)
    ix = np.tile(ids[:, None], (1, 7))
    ai = np.zeros((n, 7), dtype=np.int64)
    coeff = np.ones(n, dtype=np.int64)
    if family == "Dis":
        coeff = np.where(xyz[:, 0] < g // 2, 1, 8)
    elif family not in ("Const", "Ani"):
        raise ValueError(family)
    weights = (16, 1, 1) if family == "Ani" else (1, 1, 1)
    # Positive mass 1. Include Dirichlet boundary contributions on the diagonal.
    ai[:, 0] = 1
    for axis, stride in enumerate((g * g, g, 1)):
        for sign, slot in [(-1, 1 + 2 * axis), (1, 2 + 2 * axis)]:
            valid = xyz[:, axis] > 0 if sign == -1 else xyz[:, axis] < g - 1
            nei = np.clip(ids + sign * stride, 0, n - 1)
            w = weights[axis] * np.maximum(coeff, coeff[nei])
            w = np.where(valid, w, weights[axis] * coeff)
            ai[:, 0] += w
            ai[valid, slot] = -w[valid]
            ix[valid, slot] = nei[valid]
    ae = -int(int(ai[:, 0].max()).bit_length())
    rng = np.random.default_rng(spec["seed"])
    pool = rng.choice([-2, -1, 1, 2], size=(n, 8)).astype(np.int64)
    xx = pool[:, :s].copy()
    xe = scale
    if ce is not None and s > 1:
        if ce >= 0:
            raise ValueError("corr_exp must be negative")
        xx[:, 0] = pool[:, 0] * (1 << (-ce))
        xx[:, 1:] = pool[:, 0:1] * (1 << (-ce)) + pool[:, 1:s]
        xe = scale + ce
    # Every product and every partial sum is integral and strictly within 53 bits.
    absolute_bound = int(np.abs(ai).sum(axis=1).max()) * int(np.abs(xx).max())
    if absolute_bound >= 2**53:
        raise ValueError("Exact RHS construction range exceeded")
    bi = np.zeros_like(xx)
    for j in range(7):
        bi += ai[:, j : j + 1] * xx[ix[:, j], :]
    a = np.ldexp(ai.astype(np.float64), ae)
    xr = np.ldexp(xx.astype(np.float64), xe)
    b = np.ldexp(bi.astype(np.float64), ae + xe)
    assert np.isfinite(b).all() and np.all(np.linalg.norm(b, axis=0) > 0)
    meta = dict(
        manufactured_rhs_exact=True,
        reference_residual_norm_upper=0.0,
        reference_kind="integer dyadic exact construction; no approximate solve",
        integer_sum_absolute_bound=absolute_bound,
        A_exponent=ae,
        X_exponent=xe,
        strict_certificate=False,
    )
    return a, ix, b, xr, meta
