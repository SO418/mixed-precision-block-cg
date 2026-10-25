"""Iteration counts of scalar BJAC-PCG under small relative RHS perturbations.

Matrix: solverchallenge23_02 (n = 6291456), in the original variable ordering of the file.
BJAC uses 32 algebraic blocks, k = 2 complete corrections and t = 2 local Jacobi steps.
The outer PCG runs in FP64; the preconditioner runs in FP64 or FP32. The stopping test
is the recursive relative residual with tolerance 1e-10.

Usage:
  python3 rhs_perturbation.py --build-cache
  python3 rhs_perturbation.py --first-seed 0 --num-seeds 20 --delta 1e-14 --out runs.json

Only iteration counts are meaningful; timings from concurrent runs are not comparable.
"""

import argparse
import json
import os
import time

import numpy as np
import torch

MTX = os.environ.get("SC23_02_MTX", "/path/to/solverchallenge23_02_A.mtx")
RHS = os.environ.get("SC23_02_RHS", "/path/to/solverchallenge23_02_b.rhs")
CACHE = os.environ.get("SC23_02_CACHE", "/path/to/sc23_02_cache.npz")
NB, K_INTER, T_INTRA = 32, 2, 2
TOL = 1e-10
F64, F32 = "fp64-uniform-BJAC PCG", "fp32-fMP-BJAC PCG"


def spmv(A, x):
    return torch.sparse.mm(A, x.unsqueeze(1)).squeeze(1)


def build_cache_bd(A, nb):
    """Block-diagonal part of A and the inverse diagonal."""
    A = A.coalesce()
    n = A.shape[0]
    bs = n // nb
    idx, val = A.indices(), A.values()
    same = (idx[0] // bs) == (idx[1] // bs)
    Abd = torch.sparse_coo_tensor(
        idx[:, same], val[same], (n, n), device=A.device, dtype=A.dtype
    ).coalesce()
    diag = torch.zeros(n, device=A.device, dtype=A.dtype)
    dm = idx[0] == idx[1]
    diag[idx[0][dm]] = val[dm]
    return {"Abd": Abd, "dinv": 1.0 / diag, "nnz_bd": int(Abd._nnz())}


def apply_bjac(r, A, cache, k=K_INTER, t=T_INTRA):
    Abd, dinv = cache["Abd"], cache["dinv"]
    z = torch.zeros_like(r)
    for _ in range(k):
        res = r - spmv(A, z)
        y = torch.zeros_like(res)
        for _ in range(t):
            y = y + dinv * (res - spmv(Abd, y))
        z = z + y
    return z


@torch.no_grad()
def pcg(A64, A32, b, c64, c32, method, tol, maxit=200000):
    def prec(r):
        if method == F64:
            return apply_bjac(r, A64, c64)
        return apply_bjac(r.to(torch.float32), A32, c32).to(torch.float64)

    x = torch.zeros_like(b)
    r = b - spmv(A64, x)
    r0 = torch.linalg.norm(r).item()
    if r0 == 0.0:
        return [0.0], 0
    relres = 1.0
    hist = [relres]
    z = prec(r)
    p = z.clone()
    rz_old = torch.sum(r * z)
    for it in range(maxit):
        Ap = spmv(A64, p)
        alpha = rz_old / torch.sum(p * Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        relres = torch.linalg.norm(r).item() / r0
        hist.append(relres)
        if relres < tol:
            break
        z = prec(r)
        rz_new = torch.sum(r * z)
        p = z + (rz_new / rz_old) * p
        rz_old = rz_new
    return hist, it + 1


def build_cache():
    """Parse the matrix and RHS once and store them as .npz.

    The SolverChallenge files use 0-based indices in MatrixMarket layout.
    """
    rows, cols, vals = [], [], []
    with open(MTX) as f:
        line = f.readline()
        while line.startswith("%"):
            line = f.readline()
        n, m, nnz = map(int, line.split())
        for line in f:
            s = line.split()
            if len(s) == 3:
                rows.append(int(s[0]))
                cols.append(int(s[1]))
                vals.append(float(s[2]))
    with open(RHS) as f:
        b = np.array([float(x) for x in f.readlines()[1:]], dtype=np.float64)
    np.savez(
        CACHE,
        row=np.array(rows, dtype=np.int64),
        col=np.array(cols, dtype=np.int64),
        val=np.array(vals, dtype=np.float64),
        b=b,
        n=n,
    )
    print(f"cache written: {CACHE} (n={n}, nnz={len(vals)})", flush=True)


def load(dev):
    z = np.load(CACHE)
    n = int(z["n"])
    idx = torch.from_numpy(np.vstack([z["row"], z["col"]])).long()
    A = torch.sparse_coo_tensor(
        idx, torch.from_numpy(z["val"]), (n, n), device=dev, dtype=torch.float64
    ).coalesce()
    A = (0.5 * (A + A.t())).coalesce()  # symmetrize in FP64
    b = torch.from_numpy(z["b"]).to(dev, torch.float64)
    return A, b, n


def main():
    pa = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    pa.add_argument("--build-cache", action="store_true")
    pa.add_argument("--first-seed", type=int, default=0)
    pa.add_argument("--num-seeds", type=int, default=20)
    pa.add_argument("--delta", type=float, default=1e-14)
    pa.add_argument("--out", default="runs.json")
    pa.add_argument("--device", default="cuda:0")
    args = pa.parse_args()
    if args.build_cache:
        build_cache()
        return

    dev = args.device
    A64, b0, n = load(dev)
    A32 = A64.to(torch.float32).coalesce()
    c64, c32 = build_cache_bd(A64, NB), build_cache_bd(A32, NB)
    kept = 100 * c64["nnz_bd"] / A64._nnz()
    print(f"n={n}, block-diagonal part keeps {kept:.1f}% of the nonzeros", flush=True)

    res = {"n": n, "tol": TOL, "nb": NB, "k": K_INTER, "t": T_INTRA, "runs": []}
    seeds = range(args.first_seed, args.first_seed + args.num_seeds)
    for s in seeds:
        g = torch.Generator(device=dev)
        g.manual_seed(s)
        d = args.delta
        b = b0 * (1.0 + d * torch.randn(b0.shape, generator=g, device=dev, dtype=b0.dtype))
        rec = {"delta": d, "seed": s}
        for meth in (F64, F32):
            ts = time.time()
            h, it = pcg(A64, A32, b, c64, c32, meth, TOL)
            rec[meth] = {
                "iters": int(it),
                "relres": float(h[-1]),
                "sec": round(time.time() - ts, 2),
            }
        u, l = rec[F64]["iters"], rec[F32]["iters"]
        rec["change_pct"] = (l / u - 1) * 100
        res["runs"].append(rec)
        print(f"seed {s}: fp64 {u}, fp32 {l}, change {rec['change_pct']:+.2f}%", flush=True)
        with open(args.out, "w") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
