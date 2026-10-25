import numpy as np
from fractions import Fraction as F


def quantize(a, precision):
    a = np.asarray(a, dtype=np.float64)
    if precision == "bf16":
        f = a.astype(np.float32)
        bits = f.view(np.uint32)
        rounded = (bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)) & np.uint32(0xFFFF0000)
        rounded = np.where(np.isfinite(f), rounded, bits).astype(np.uint32)
        return rounded.view(np.float32).astype(np.float64)
    return a.astype({"fp16": np.float16, "fp32": np.float32, "fp64": np.float64}[precision]).astype(
        np.float64
    )


def conversion_model(precision):
    if precision == "fp64":
        return F(0), F(0)
    if precision == "fp16":
        return F(1, 2**11), F(1, 2**25)
    if precision == "fp32":
        return F(1, 2**24), F(1, 2**150)
    if precision == "bf16":
        u1, u2 = F(1, 2**24), F(1, 2**8)
        return u1 + u2 + u1 * u2, (1 + u2) * F(1, 2**150) + F(1, 2**134)
    raise ValueError(precision)
