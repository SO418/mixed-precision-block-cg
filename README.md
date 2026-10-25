# Mixed-Precision QR-Based Block CG

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-2.0.2-013243?logo=numpy&logoColor=white)
![CuPy](https://img.shields.io/badge/CuPy-14.0.1-2E8B57)
![CUDA](https://img.shields.io/badge/CUDA-13.0-76B900?logo=nvidia&logoColor=white)

Code and records for the numerical experiments in *Rounding Error Analysis of Mixed-Precision
QR-Based Block Conjugate Gradients*. The outer iteration runs in FP64; the block-Jacobi (BJAC)
preconditioner runs in FP64, FP32, FP16 or BF16.

## Layout

```
exact/                  # reference BJAC + QR block CG, bounds checked in exact rational arithmetic
cpu_three_matrices/     # three matrices x four formats; 108-configuration set; block width study
cpu_interval16/         # order-16 example with the finite-interval bound
cpu_controlled/         # small controlled checks
  precision_selection.py  # all 16 FP16/FP32 block assignments
  scalar_affine.py        # conventional PCG and the input-independent affine BJAC bound
  outer_budget.py         # preconditioner vs. outer error bounds
gpu_bounds/             # bound comparisons on a native GPU (n = 110592)
gpu_cost/               # GPU accuracy and cost: FP16, scaling, precision switching
rhs_perturbation/       # iteration counts under small RHS perturbations
data/                   # records used in the paper, same layout as the code
```

## Experiments

### [`exact`](exact) — reference implementation and certification

Rounded BJAC preconditioning inside QR-based block CG, plus the bound monitor and an
independent validator.

- **Monitor**: builds the recurrence bound from the stored trajectory, primitive rounding
  bounds and analytic spectral estimates.
- **Validator**: compares every bound with the exact error from $P$ and $A^{-1}$, with no
  tolerances.

Run directly, it reproduces the nine $n = 12$ cases of the verification section in the supplement:

```bash
cd exact
python3 verify_qr_bjac.py --output traces
python3 certify_budgets.py --input traces --output certificates
```

### [`cpu_three_matrices`](cpu_three_matrices) — three matrices, four formats

Tridiagonal matrices of order 32 with couplings $a = 1/4, 3/8, 7/16, 15/32$, RHS widths 1, 4
and 8, near-correlated and scaled inputs, seeds 301–303. Each run takes 60 updates and is
certified in exact arithmetic.

```bash
cd cpu_three_matrices
python3 run_experiments.py --phase main --out results   # 108 configurations, about 3 h on one core
```

### [`cpu_interval16`](cpu_interval16) — finite-interval bound

Order-16 matrix, four inner formats, 50 updates, with the bound of the finite-interval theorem
on updates 26–50.

```bash
cd cpu_interval16
python3 self_test.py && python3 run.py --out results/exact && python3 plot.py
```

### [`cpu_controlled`](cpu_controlled) — controlled checks

- **`precision_selection.py`**: four independent $2\times2$ blocks with $d = (1, 4, 16, 64)$.
  Enumerates all 16 FP16/FP32 assignments and finds the fewest FP32 blocks meeting each target.
- **`scalar_affine.py`**: conventional PCG on an order-16 matrix (seeds 301–303, four formats)
  with its forward-error bound, and the input-independent affine bound tested at three input scales.
- **`outer_budget.py`**: four $n = 48$ cases separating the BJAC call-error bound from the
  total recurrence bound.

```bash
cd cpu_controlled
python3 precision_selection.py --out results
python3 scalar_affine.py --out results
python3 outer_budget.py --out results
```

### [`gpu_bounds`](gpu_bounds) — bounds on a native GPU

Seven-point problems on a $48^3$ grid (Const, Dis, Ani), four inner formats, up to 200 updates.
Every operation is rounded to the target format by custom CUDA kernels.

```bash
cd gpu_bounds
PYTHONPATH=$PWD python3 run_gpu.py --phase main --out results
```

> CuPy compiles kernels with `-ftz=true`, which flushes FP32 subnormals. `sitecustomize.py`
> switches this off when the directory is on `PYTHONPATH`; the self-test fails without it.

### [`gpu_cost`](gpu_cost) — accuracy and cost on the GPU

Unscaled FP16, call scaling, local scaling and precision switching against FP32 and FP64
baselines, plus isolated BJAC calls at several input scales.

```bash
cd gpu_cost
python3 calibrate.py --out results/calibrate
python3 run.py --preset core --out results/core
python3 run.py --preset scales --out results/scales
```

### [`rhs_perturbation`](rhs_perturbation) — iteration counts

Scalar BJAC–PCG on `solverchallenge23_02` ($n = 6291456$) with 20 right-hand sides perturbed
at relative amplitude $10^{-14}$, comparing FP64 and FP32 preconditioning. The matrix is used in
the ordering of the distributed file (all R, then all E, then all I); the files (SolverChallenge
2023, 0-based indices) are not included.

```bash
cd rhs_perturbation
export SC23_02_MTX=/path/to/solverchallenge23_02_A.mtx
export SC23_02_RHS=/path/to/solverchallenge23_02_b.rhs
export SC23_02_CACHE=/path/to/sc23_02_cache.npz
python3 rhs_perturbation.py --build-cache
python3 rhs_perturbation.py --first-seed 0 --num-seeds 20 --delta 1e-14 --out runs.json
```

## Requirements

```bash
pip install -r requirements.txt     # numpy 2.0.2, mpmath 1.3.0, matplotlib
pip install cupy-cuda13x            # GPU experiments
pip install torch                   # rhs_perturbation only
```

The CPU records were produced with numpy 2.0.2 and mpmath 1.3.0. Other LAPACK builds can change
the QR factors in the last bits, so traces need not match bitwise, but every check still applies.
Set `OPENBLAS_NUM_THREADS=1` for the CPU runs to keep the reduction order fixed. The GPU records
come from an NVIDIA B300.

## Data

[`data/`](data) holds the records behind the tables and figures, organized like the code; see
[`data/README.md`](data/README.md). The full CPU traces (about 160 MB compressed) are
attached to the [v1.0 release](https://github.com/SO418/mixed-precision-block-cg/releases/download/v1.0/bcg_data_full.tgz) as `bcg_data_full.tgz`.
