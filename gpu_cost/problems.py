"""Diff3D test matrices with O(n) storage.

The matrix is symmetrically Jacobi-scaled; the scaled matrix is the test problem.
"""

import numpy as np


def diff3d(g=16, kind="Const", contrast=1000.0, seed=0):
    if g < 2 or contrast <= 0:
        raise ValueError("g>=2 and contrast>0 required")
    n = g**3
    h = 1.0 / (g + 1)
    ids = np.arange(n).reshape(g, g, g)
    kap = np.ones((g, g, g))
    anis = (1.0, 1.0, 1.0)
    if kind == "Ani":
        anis = (1.0, contrast, contrast)
    elif kind == "Dis":
        c = (np.arange(g) + 1) * h
        mask = (c >= 0.25) & (c <= 0.75)
        kap[np.ix_(mask, mask, mask)] = contrast
    elif kind == "Rand":
        kap = contrast ** np.random.default_rng(seed).random((g, g, g))
    elif kind != "Const":
        raise ValueError(kind)
    vals = np.zeros((7, n))
    cols = np.zeros((7, n), dtype=np.int64)
    diag = np.zeros(n)
    for axis, (offset, plus, minus) in enumerate([(g * g, 6, 0), (g, 5, 1), (1, 4, 2)]):
        sl0 = [slice(None)] * 3
        sl1 = list(sl0)
        sl0[axis] = slice(None, -1)
        sl1[axis] = slice(1, None)
        sl0, sl1 = tuple(sl0), tuple(sl1)
        ii, jj = ids[sl0].ravel(), ids[sl1].ravel()
        aa, bb = kap[sl0].ravel(), kap[sl1].ravel()
        face = anis[axis] * (2.0 * aa * bb / (aa + bb)) / h**2
        vals[plus, ii] = -face
        cols[plus, ii] = jj
        vals[minus, jj] = -face
        cols[minus, jj] = ii
        diag[ii] += face
        diag[jj] += face
        for edge in (0, -1):
            sl = [slice(None)] * 3
            sl[axis] = edge
            sl = tuple(sl)
            diag[ids[sl].ravel()] += anis[axis] * kap[sl].ravel() / h**2
    vals[3] = diag
    cols[3] = np.arange(n)
    scale = np.sqrt(diag)
    vals /= scale[None, :] * scale[cols]
    # Scaled diffusion has A scale >= 0 and scale > 0. This supplies a
    # Collatz bound for the nonnegative H and each within-block Jacobi G.
    # Ellipticity + Dirichlet Laplacian bound before congruence scaling.
    kmin = float(kap.min()) * min(anis)
    lmin = kmin * (12.0 / h**2) * np.sin(np.pi / (2 * (g + 1))) ** 2 / diag.max()
    return vals, cols, scale, float(lmin)
