"""Problem setup on the CPU and block-selection heuristics (scores are not bounds)."""

import numpy as np
from problems import diff3d


def problem(g, kind, contrast, bs):
    vals, cols, scale, lower = diff3d(g, kind, contrast)
    vals[3] = 1.0  # Define the stored unit-diagonal problem explicitly.
    a = np.ascontiguousarray(vals.T)
    ix = np.ascontiguousarray(cols.T, dtype=np.int32)
    n = len(a)
    if n % bs:
        raise ValueError("g**3 must be divisible by block size")
    return a, ix


def block_scores(a, ix, bs, t):
    # Spectral estimates describe local amplification; no outward rounding.
    blocks = []
    for beg in range(0, len(a), bs):
        z = np.zeros((bs, bs))
        for row in range(bs):
            for v, col in zip(a[beg + row], ix[beg + row]):
                if beg <= col < beg + bs:
                    z[row, col - beg] += v
        blocks.append(z)
    z = np.array(blocks)
    theta = np.max(np.abs(1 - np.linalg.eigvalsh(z)), axis=1)
    if np.any(theta >= 1):
        raise ValueError("Local Jacobi contraction estimate unavailable")
    geom = sum(theta**j for j in range(t))
    smin = (1 - theta**t) / (1 + theta)
    # Proxy suggested by the local L_i structure; not a derived error bound.
    amp = geom / np.sqrt(smin)
    out = {}
    for bits, dtype, u in [(16, np.float16, 2**-11), (32, np.float32, 2**-24)]:
        quant = np.linalg.norm(z.astype(dtype).astype(float) - z, axis=(1, 2))
        out[bits] = amp * (7 * t * u * (1 + (1 + theta) * geom) + quant * geom)
    return out, theta


def select(scores, policy, target, seed=17):
    lo, hi = scores[16], scores[32]
    n = len(lo)
    if policy == "fp16":
        mask = np.ones(n, bool)
    elif policy in ("fp32", "fp64"):
        mask = np.zeros(n, bool)
    elif policy == "worst":
        mask = np.full(n, float(np.max(lo)) <= target)
    elif policy in ("weighted", "random_matched"):
        mask = np.ones(n, bool)
        total = float(lo @ lo)
        for j in np.argsort(-(lo * lo - hi * hi)):
            if total <= n * target * target:
                break
            total += float(hi[j] ** 2 - lo[j] ** 2)
            mask[j] = False
        if policy == "random_matched":
            count = int(mask.sum())
            mask[:] = False
            mask[np.random.default_rng(seed).permutation(n)[:count]] = True
    else:
        raise ValueError(policy)
    predicted = np.where(mask, lo, hi)
    return mask, {
        "fp16_fraction": float(mask.mean()),
        "proxy_rms": float(np.sqrt(np.mean(predicted**2))),
        "proxy_max": float(predicted.max()),
        "proxy_target_met": bool(np.sqrt(np.mean(predicted**2)) <= target),
        "certificate": False,
    }


def dense(a, ix):
    z = np.zeros((len(a), len(a)))
    for i in range(len(a)):
        for v, j in zip(a[i], ix[i]):
            z[i, j] += v
    return z


def cpu_local(a, ix, rhs, bs, t, mask, dtype_global=np.float32):
    out = np.empty_like(rhs, dtype=dtype_global)
    for beg in range(0, len(a), bs):
        dtype = np.float16 if mask[beg // bs] else dtype_global
        aa = a.astype(dtype)
        r = rhs[beg : beg + bs].astype(dtype)
        u = np.zeros_like(r)
        for _ in range(t):
            w = np.zeros_like(r)
            for i in range(bs):
                for j in range(7):
                    col = ix[beg + i, j]
                    if beg <= col < beg + bs and aa[beg + i, j] != 0:
                        w[i] = dtype(w[i] + dtype(aa[beg + i, j] * u[col - beg]))
            u = dtype(u + dtype(r - w))
        out[beg : beg + bs] = u
    return out
