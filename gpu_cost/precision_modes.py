"""Precision variants: call scaling, local scaling and switching.

The sparse and Jacobi kernels and the outer solver are those of gpu.py.
"""

import numpy as np

METHODS = ("fp64", "fp32", "fp16", "call_scaled", "local_scaled", "switch")


def normalize_columns(x, xp):
    """Return 2**(-e) * x with column max in [0.5,1), and integer e.
    frexp(0) gives e=0. ldexp avoids materializing overflowing scale factors.
    """
    maximum = xp.max(xp.abs(x), axis=0)
    _, exponent = xp.frexp(maximum)
    exponent = exponent.astype(xp.int32)
    return xp.ascontiguousarray(xp.ldexp(x, -exponent[None, :])), exponent


def host(x, xp):
    return x if xp is np else xp.asnumpy(x)


class PrecisionBJAC:
    def __init__(self, k, a, ix, bs, t, p, method, bnorm, xp, threshold=1e-4, diagnostics=False):
        if method not in METHODS:
            raise ValueError(method)
        if bs <= 0 or t < 1 or p < 1 or len(a) % bs:
            raise ValueError("invalid block/step counts")
        if not 0 < threshold < 1:
            raise ValueError("threshold must be in (0,1)")
        self.k, self.ix, self.bs, self.t, self.p, self.method, self.xp = k, ix, bs, t, p, method, xp
        self.threshold = threshold
        self.bnorm = bnorm
        self.diagnostics = diagnostics
        self.calls = 0
        self.half_calls = 0
        self.single_calls = 0
        self.double_calls = 0
        self.switched = False
        self.switch_call = None
        self.switch_ratios = None
        self.trace = []
        self.a = {64: a} if method == "fp64" else {32: a.astype(xp.float32)}
        if method not in ("fp64", "fp32"):
            self.a[16] = a.astype(xp.float16)
        self.ids = xp.arange(len(a) // bs, dtype=xp.int32)

    def _record(self, physical, used, bits, j, outer_e, local_e):
        xp = self.xp
        dtype = {16: xp.float16, 32: xp.float32, 64: xp.float64}[bits]
        if outer_e is None:
            outer_e = xp.zeros(used.shape[1], dtype=xp.int32)
        if local_e is None:
            local_e = xp.zeros(used.shape[1], dtype=xp.int32)

        def zero_loss(v):
            nonzero = xp.sum(v != 0, axis=0)
            count = xp.sum((v != 0) & (v.astype(dtype) == 0), axis=0)
            return host(count / xp.maximum(nonzero, 1), xp).tolist()

        self.trace.append(
            dict(
                call=self.calls + 1,
                correction=j + 1,
                bits=bits,
                physical_input_maxabs=host(xp.max(xp.abs(physical), axis=0), xp).tolist(),
                kernel_input_maxabs=host(xp.max(xp.abs(used), axis=0), xp).tolist(),
                zero_loss_without_scaling=zero_loss(physical),
                zero_loss_at_kernel=zero_loss(used),
                call_scale_exponent=host(outer_e, xp).tolist(),
                local_scale_exponent=host(local_e, xp).tolist(),
            )
        )

    def __call__(self, r):
        xp = self.xp
        if self.method == "switch" and not self.switched:
            # Protect the first column entering the small-residual regime.
            # One-way upgrade of all blocks and columns; no later downgrade and no restart.
            ratios = xp.linalg.norm(r, axis=0) / self.bnorm
            if bool(host(xp.any(ratios <= self.threshold), xp)):
                self.switched = True
                self.switch_call = self.calls + 1
                self.switch_ratios = host(ratios, xp).tolist()
        bits = (
            64 if self.method == "fp64" else (32 if self.method == "fp32" or self.switched else 16)
        )
        dtype = xp.float64 if bits == 64 else xp.float32
        outer_e = None
        if self.method == "call_scaled":
            r, outer_e = normalize_columns(r, xp)
        rhs = xp.ascontiguousarray(r, dtype=dtype)
        y = xp.zeros_like(rhs)
        for j in range(self.p):
            residual = rhs - self.k.sp(self.a[64 if bits == 64 else 32], self.ix, y)
            local_e = None
            used = residual
            if self.method == "local_scaled":
                used, local_e = normalize_columns(residual, xp)
            used = xp.ascontiguousarray(used)
            if self.diagnostics:
                physical = residual.astype(xp.float64)
                if outer_e is not None:
                    physical = xp.ldexp(physical, outer_e[None, :])
                self._record(physical, used, bits, j, outer_e, local_e)
            corr = self.k.local(self.a, self.ix, used, self.bs, self.t, {bits: self.ids})
            if self.method == "local_scaled":
                corr = xp.ldexp(corr, local_e[None, :])
            y = y + corr
        self.calls += 1
        if bits == 16:
            self.half_calls += 1
        elif bits == 32:
            self.single_calls += 1
        else:
            self.double_calls += 1
        out = y.astype(xp.float64)
        if self.method == "call_scaled":
            out = xp.ldexp(out, outer_e[None, :])
        return out

    def statistics(self):
        return dict(
            calls=self.calls,
            half_calls=self.half_calls,
            single_calls=self.single_calls,
            double_calls=self.double_calls,
            switch_call=self.switch_call,
            switch_ratios=self.switch_ratios,
            threshold=self.threshold if self.method == "switch" else None,
        )
