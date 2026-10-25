"""Separate rounding of each multiply and add on NumPy or CuPy, with error bounds.

The bounds are evaluated in ordinary FP64 (no directed rounding).
"""

import math
import numpy as np

FORMATS = {
    "fp64": (2.0**-53, 2.0**-1074),
    "fp32": (2.0**-24, 2.0**-150),
    "fp16": (2.0**-11, 2.0**-25),
    # Conversion via FP32, then BF16: account for both roundings.
    "bf16": (2.0**-8 + 2.0**-24 + 2.0**-32, 2.0**-134 + 2.0**-149),
}


class Backend:
    def __init__(self, device="cpu"):
        if device == "cuda":
            import cupy as cp

            self.xp = cp
            cp.cuda.runtime.getDeviceCount()
        elif device == "cpu":
            self.xp = np
        else:
            raise ValueError(device)
        self.device = device

    def array(self, x):
        return self.xp.asarray(x, dtype=self.xp.float64)

    def host(self, x):
        return np.asarray(x) if self.device == "cpu" else self.xp.asnumpy(x)

    def scalar(self, x):
        return float(x)

    def norm(self, x):
        return float(self.xp.linalg.norm(x))

    def sync(self):
        if self.device == "cuda":
            self.xp.cuda.get_current_stream().synchronize()

    def q(self, x, fmt):
        xp = self.xp
        if fmt == "fp64":
            return x.astype(xp.float64, copy=False)
        if fmt == "bf16":
            f = x.astype(xp.float32)
            bits = f.view(xp.uint32)
            # RNE, retaining NaN/Inf rather than wrapping their payloads.
            rounded = (bits + xp.uint32(0x7FFF) + ((bits >> 16) & 1)) & xp.uint32(0xFFFF0000)
            rounded = xp.where(xp.isfinite(f), rounded, bits).astype(xp.uint32)
            return rounded.view(xp.float32).astype(xp.float64)
        return x.astype({"fp16": xp.float16, "fp32": xp.float32}[fmt]).astype(xp.float64)

    def env(self, stored, fmt, conversion=False):
        u, tau = FORMATS[fmt]
        if fmt == "fp64" and conversion:
            return self.xp.zeros_like(stored)
        if not conversion and fmt != "fp64":
            u = u + 2.0**-53 + u * 2.0**-53
            tau = tau + (1 + u) * 2.0**-1074
        return (u * self.xp.abs(stored) + tau) / (1 - u)

    def cast(self, x, fmt):
        v = self.q(x, fmt)
        return v, self.env(v, fmt, True)

    def op(self, a, b, kind, fmt):
        raw = {"add": lambda: a + b, "sub": lambda: a - b, "mul": lambda: a * b}[kind]()
        out = self.q(raw, fmt)
        return out, self.env(out, fmt)


class Matrix:
    """ELLPACK; diagonal slot 0, no repeated nonzero indices in a row."""

    def __init__(self, a, ix, backend, bs, t, p):
        self.be = backend
        self.xp = backend.xp
        self.a = backend.array(a)
        self.ix = self.xp.asarray(ix, dtype=self.xp.int32)
        self.n = len(a)
        self.width = a.shape[1]
        self.bs = bs
        self.t = t
        self.p = p
        if self.n % bs:
            raise ValueError("n must be divisible by block size")
        self.nb = self.n // bs
        self.diag = self.a[:, 0:1]
        self.local = self.ix // bs == (self.xp.arange(self.n) // bs)[:, None]
        # For symmetric positive M matrices. Weighted Gershgorin in D^{-1/2}AD^{-1/2}.
        da = np.asarray(a[:, 0])
        normal = a / np.sqrt(da[:, None] * da[ix])
        off = np.sum(np.abs(normal[:, 1:]), axis=1)
        direct_off = np.sum(np.abs(a[:, 1:]), axis=1)
        self.amin = float(max(np.min(da - direct_off), np.min(da) * np.min(1 - off)))
        self.aplus = float(min(np.max(da + direct_off), np.max(da) * np.max(1 + off)))
        if self.amin <= 0:
            raise ValueError("No positive spectral enclosure for this input")
        qrows = np.sum(np.abs(normal[:, 1:]) * backend.host(self.local[:, 1:]), axis=1)
        direct_q = np.sum(np.abs(a[:, 1:]) * backend.host(self.local[:, 1:]), axis=1) / da
        self.rhoi = np.minimum(
            qrows.reshape(self.nb, bs).max(axis=1), direct_q.reshape(self.nb, bs).max(axis=1)
        )
        if np.max(self.rhoi) >= 1:
            raise ValueError("Local Jacobi condition unavailable")
        self.smin = (1 - self.rhoi**t) / (1 + self.rhoi)
        self.bplus_i = np.array(
            [
                sum(q**j for j in range(t)) / da[k * bs : (k + 1) * bs].min()
                for k, q in enumerate(self.rhoi)
            ]
        )
        self.bplus = float(self.bplus_i.max())
        # H=J_local^t + B_local * (-A_off) is nonnegative.
        # Use H*v/v directly to avoid cancellation in v-B*A*v.
        candidates = []
        for v in [self.xp.ones_like(self.diag), 1 / self.xp.sqrt(self.diag)]:
            power = v.copy()
            for _ in range(t):
                off_local = self.mv_raw(power, local=True, offdiag=True)
                power = -off_local / self.diag
            off_rhs = -self.mv_raw(v, offblock=True)
            corr = self.local_reference(off_rhs)
            candidates.append(float(self.xp.max((power + corr) / v)))
        self.rho = min(candidates)
        self.chi = self.rho**p
        self.nu = 1.0 if p % 2 == 0 else 1 + self.chi
        self.betab = 1 + self.rho
        self.condition_ok = bool(0 <= self.rho < 1)

    def mv_raw(self, x, local=False, offdiag=False, offblock=False, absolute=False):
        xp = self.xp
        y = xp.zeros_like(x)
        for j in range(1 if offdiag else 0, self.width):
            a = self.a[:, j : j + 1]
            if local:
                a = a * self.local[:, j : j + 1]
            if offblock:
                a = a * (~self.local[:, j : j + 1])
            if absolute:
                a = xp.abs(a)
            y = y + a * x[self.ix[:, j], :]
        return y

    def mv(self, x):
        # Separate FP64 products and additions, no BLAS sparse reduction ambiguity.
        return self.mv_raw(x)

    def local_reference(self, s):
        u = self.xp.zeros_like(s)
        for _ in range(self.t):
            u = u + (s - self.mv_raw(u, local=True)) / self.diag
        return u

    def dense(self):
        a = np.zeros((self.n, self.n))
        aa = self.be.host(self.a)
        ii = self.be.host(self.ix)
        for i in range(self.n):
            for j in range(self.width):
                a[i, ii[i, j]] += aa[i, j]
        return a

    def constants(self):
        return {
            k: getattr(self, k)
            for k in ["amin", "aplus", "rho", "chi", "nu", "betab", "bplus", "condition_ok"]
        }


class BJAC:
    def __init__(self, m, global_fmt="fp32", block_formats=None, monitor=True):
        self.m = m
        self.be = m.be
        self.xp = m.xp
        self.gf = global_fmt
        self.fmts = block_formats or [global_fmt] * m.nb
        if len(self.fmts) != m.nb:
            raise ValueError("one local format required per block")
        self.groups = {
            f: self.xp.asarray(np.repeat(np.array(self.fmts) == f, m.bs)) for f in set(self.fmts)
        }
        self.ag = self.be.q(m.a, global_fmt)
        self.al = m.a.copy()
        self.dinv = self.xp.empty_like(m.diag)
        for f, rows in self.groups.items():
            self.al[rows] = self.be.q(m.a[rows], f)
            self.dinv[rows] = self.be.q(1 / m.diag[rows], f)
        self.pi = self.xp.abs(self.dinv - 1 / m.diag) + 2.0**-52 * self.xp.abs(1 / m.diag)
        self.monitor = monitor

    def local_cast(self, x):
        v = self.xp.empty_like(x)
        e = self.xp.empty_like(x)
        for f, rows in self.groups.items():
            v[rows], e[rows] = self.be.cast(x[rows], f)
        return v, e

    def local_op(self, a, b, kind):
        v = self.xp.empty_like(a)
        e = self.xp.empty_like(a)
        for f, rows in self.groups.items():
            v[rows], e[rows] = self.be.op(a[rows], b[rows], kind, f)
        return v, e

    def mm(self, x, local=False):
        xp = self.xp
        m = self.m
        aa = self.al if local else self.ag
        y = xp.zeros_like(x)
        error = xp.zeros_like(x)
        op = self.local_op if local else lambda a, b, k: self.be.op(a, b, k, self.gf)
        for j in range(m.width):
            mask = m.local[:, j : j + 1] if local else 1
            a = aa[:, j : j + 1] * mask
            xx = x[m.ix[:, j], :]
            prod, ep = op(xp.broadcast_to(a, x.shape), xx, "mul")
            y, ea = op(y, prod, "add")
            error += ep + ea + xp.abs((aa[:, j : j + 1] - m.a[:, j : j + 1]) * mask) * xp.abs(xx)
        return y, error

    def __call__(self, r):
        xp = self.xp
        m = self.m
        be = self.be
        rhs, ecast = be.cast(r, self.gf)
        y = xp.zeros_like(r)
        shape = (m.nb, m.bs, r.shape[1])
        rhoi = be.array(m.rhoi)[:, None]
        smin = be.array(m.smin)[:, None]
        eta_cast = m.nu / math.sqrt(m.amin) * xp.linalg.norm(ecast, axis=0)
        propagated = xp.zeros(r.shape[1])
        maxcj = xp.zeros_like(propagated)
        # coarse: replace block contributions by maximum, then correction contributions by maximum.
        conv_coarse = xp.zeros_like(propagated)
        maxcj_coarse = xp.zeros_like(propagated)
        cjs = []
        castzero = int(xp.count_nonzero((r != 0) & (rhs == 0)))
        block_parts = None
        for sw in range(m.p):
            ay, ew = self.mm(y)
            residual, er = be.op(rhs, ay, "sub", self.gf)
            eres = ew + er
            s, es = self.local_cast(residual)
            castzero += int(xp.count_nonzero((residual != 0) & (s == 0)))
            u = xp.zeros_like(r)
            acc = xp.zeros((m.nb, r.shape[1]))
            for _ in range(m.t):
                w, wg = self.mm(u, True)
                v, vg = self.local_op(s, w, "sub")
                c, cg = self.local_op(xp.broadcast_to(self.dinv, r.shape), v, "mul")
                un, ug = self.local_op(u, c, "add")
                dq = (
                    xp.abs(self.dinv) * wg
                    + self.pi * (xp.abs(s - w) + wg)
                    + xp.abs(self.dinv) * vg
                    + cg
                    + ug
                )
                dn = xp.sqrt(xp.sum((xp.sqrt(m.diag) * dq).reshape(shape) ** 2, axis=1))
                acc = rhoi * acc + dn
                u = un
            li = acc / xp.sqrt(smin)
            li += xp.sqrt(be.array(m.bplus_i))[:, None] * xp.linalg.norm(es.reshape(shape), axis=1)
            # Return local correction into the global arithmetic; bound the transfer too.
            corr, etrans = be.cast(u, self.gf)
            yn, eu = be.op(y, corr, "add", self.gf)
            cr = math.sqrt(m.betab * m.bplus) * xp.linalg.norm(eres, axis=0)
            ci = math.sqrt(m.betab) * xp.linalg.norm(li, axis=0)
            cu = math.sqrt(m.aplus) * xp.linalg.norm(eu + etrans, axis=0)
            cj = cr + ci + cu
            coarse = cr + math.sqrt(m.betab * m.nb) * xp.max(li, axis=0) + cu
            propagated = m.rho * propagated + cj
            conv_coarse = m.rho * conv_coarse + coarse
            maxcj = xp.maximum(maxcj, cj)
            maxcj_coarse = xp.maximum(maxcj_coarse, coarse)
            y = yn
            cjs.append(be.host(cj).tolist())
            block_parts = li
        eta = eta_cast + propagated
        geom = sum(m.rho**j for j in range(m.p))
        coarse = eta_cast + geom * maxcj_coarse
        finite = bool(xp.all(xp.isfinite(y)))
        return y, {
            "eta": be.norm(eta),
            "eta_columns": be.host(eta).tolist(),
            "coarse_eta": be.norm(coarse),
            "cast_part": be.norm(eta_cast),
            "rounding_part": be.norm(propagated),
            "correction_cj": cjs,
            "last_block_L": be.host(block_parts).tolist(),
            "input_zero_loss_count": castzero,
            "finite": finite,
            "condition_ok": m.condition_ok,
            "strict_certificate": False,
        }


def dot_env(a, b, xp):
    """Standard error bound for an FP64 reduction of length n (sequential, tree or FMA).

    All outer operands are FP64, so TF32 is never involved.
    """
    n = a.shape[1]
    u = 2.0**-53
    gamma = (2 * n + 2) * u / (1 - (2 * n + 2) * u)
    return gamma * (xp.abs(a) @ xp.abs(b)) + (2 * n + 2) * 2.0**-1074


def add_env(a, b, xp):
    return 2.0**-53 * (xp.abs(a) + xp.abs(b)) + 2.0**-1074


def sparse_env(m, x):
    n = m.width
    u = 2.0**-53
    return ((2 * n + 2) * u / (1 - (2 * n + 2) * u)) * m.mv_raw(m.xp.abs(x), absolute=True) + (
        2 * n + 2
    ) * 2.0**-1074
