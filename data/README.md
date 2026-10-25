# Experiment records

Records behind the numbers and figures of the paper. Directory names match the code
directories at the top of this repository. Platform strings and absolute paths in the environment files were
redacted; nothing else was edited.

| Directory | Contents |
|---|---|
| `certification/` | Nine n = 12 cases: stored traces and exact rational certificates (`summary.json`, `SUMMARY.md`). The certificates were regenerated from the traces with the code in `bcg_code/exact`. |
| `cpu_three_matrices/` | The 108-configuration CPU set (`main/`) plus the smoke and pilot phases. Each case has `config.json`, `result.json` and `metrics.csv`; `summary.json` lists all cases. `figure_inputs/` holds the three matrices, the shared right-hand side and the final iterates used in the three-matrix figure (case ids in `summary.json`). `block_study/` holds the block-width and correlation summaries. |
| `cpu_interval16/` | Order-16 example, four inner formats, 50 updates: per-run results and summary. |
| `cpu_controlled/` | Blockwise precision assignments (`precision_selection.json`), outer-bound cases (`outer_summary.json`, `outer_cases/`), conventional PCG (`scalar_*`) and the affine call bound (`affine_summary.json`). |
| `gpu_bounds/` | Native GPU runs (smoke, pilot, main; one JSON per run in `runs/`). The paper uses the g = 48 runs of `main/`. `first_attempt_without_ftz_fix/` is the self-test that failed before the flush-to-zero fix. |
| `gpu_cost/` | `core/` gives the GPU cost table, `core_diagnostics/` the residual histories (seed 301), `scales/` the RHS scaling test, `calibrate/` the isolated BJAC calls. `environment_lock.txt` lists the Python packages. |
| `rhs_perturbation/` | Iteration counts for the 20 perturbed right-hand sides (seeds 0-19, relative amplitude 1e-14). Timings came from concurrent runs and are not comparable. |

The full CPU traces, certificates and validation files (about 1 GB uncompressed) are in
[`bcg_data_full.tgz`](https://github.com/SO418/mixed-precision-block-cg/releases/download/v1.0/bcg_data_full.tgz) (v1.0 release), with the same layout.
