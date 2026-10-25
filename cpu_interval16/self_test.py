import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "exact"))
import numpy as np
from fractions import Fraction as F
from formats import quantize
from certify_budgets import LowMonitor, fm

for fmt in ["fp64", "fp32", "fp16", "bf16"]:
    mon = LowMonitor(fmt)
    a = np.array([[0.0, 1.0, -1.0, 2.0**-20, 1.00390625, 1.01171875]])
    mon.cast(a)
    for kind in ["+", "-", "*", "/"]:
        mon.op(a, np.full_like(a, 1.5), kind)
assert np.array_equal(quantize(np.array([1.00390625, 1.01171875]), "bf16"), [1.0, 1.015625])
assert np.array_equal(
    quantize(np.array([2.0**-133, 2.0**-134, 3 * 2.0**-134]), "bf16"), [2.0**-133, 0.0, 2.0**-132]
)
mon = LowMonitor("bf16")
mon.cast(np.array([[2.0**-133, 2.0**-134, 3 * 2.0**-134]]))
print("PASS: four-format primitive envelopes, BF16 ties and subnormals")
