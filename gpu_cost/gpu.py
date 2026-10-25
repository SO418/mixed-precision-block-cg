"""CUDA sparse products and shared-memory Jacobi kernels.

FP16 uses __hmul/__hadd/__hsub with half-precision shared memory.
"""

from pathlib import Path
import time
import numpy as np
import cupy as cp
import cupyx

HEADER = r"""
#include <cuda_fp16.h>
template<typename T> __device__ T plus_(T a,T b){return a+b;}
template<typename T> __device__ T minus_(T a,T b){return a-b;}
template<typename T> __device__ T times_(T a,T b){return a*b;}
template<> __device__ half plus_(half a,half b){return __hadd(a,b);}
template<> __device__ half minus_(half a,half b){return __hsub(a,b);}
template<> __device__ half times_(half a,half b){return __hmul(a,b);}
"""


def source():
    src = HEADER
    for name, T in [("sp32", "float"), ("sp64", "double")]:
        src += f"""
extern "C" __global__ void {name}(const {T}* a,const int* ix,const {T}* x,{T}* y,int n,int s){{
 int k=blockIdx.x*blockDim.x+threadIdx.x;
 if(k>=n*s)return;
 int row=k/s, col=k%s; {T} acc=0;
 for(int j=0;j<7;j++){{{T} v=a[row*7+j]; if(v!=0) acc=plus_(acc,times_(v,x[ix[row*7+j]*s+col]));}}
 y[k]=acc;
}}
"""
    for name, T, O in [
        ("local16", "half", "float"),
        ("local32", "float", "float"),
        ("local64", "double", "double"),
    ]:
        src += f"""
extern "C" __global__ void {name}(const {T}* a,const int* ix,const {O}* r,{O}* corr,
 const int* blocks,int bs,int s,int steps){{
 extern __shared__ unsigned char memory[];
 {T}* u=({T}*)memory; {T}* un=u+bs*s;
 int lane=threadIdx.x, row=lane/s, col=lane%s;
 int beg=blocks[blockIdx.x]*bs, global=beg+row;
 u[lane]=({T})0; __syncthreads();
 {T} rhs=({T})r[global*s+col];
 for(int it=0;it<steps;it++){{
  {T} acc=({T})0;
  for(int j=0;j<7;j++){{
   int nb=ix[global*7+j]; {T} val=a[global*7+j];
   if(nb>=beg && nb<beg+bs && (double)val!=0.0)
      acc=plus_(acc,times_(val,u[(nb-beg)*s+col]));
  }}
  un[lane]=plus_(u[lane],minus_(rhs,acc));
  __syncthreads(); {T}* tmp=u;u=un;un=tmp; __syncthreads();
 }}
 corr[global*s+col]=({O})u[lane];
}}
"""
    return src


class Kernels:
    def __init__(self):
        self.module = cp.RawModule(
            code=source(), options=("--std=c++14", "--fmad=false", "--ftz=false")
        )
        self.fun = {
            n: self.module.get_function(n)
            for n in ("sp32", "sp64", "local16", "local32", "local64")
        }

    def sp(self, a, ix, x):
        x = cp.ascontiguousarray(x)
        n, s = x.shape
        y = cp.empty_like(x)
        self.fun["sp64" if a.dtype == cp.float64 else "sp32"](
            ((n * s + 255) // 256,), (256,), (a, ix, x, y, np.int32(n), np.int32(s))
        )
        return y

    def local(self, a, ix, r, bs, t, mask, global64=False):
        n, s = r.shape
        if bs * s > 1024:
            raise ValueError("block_size * RHS must be <= 1024")
        out = cp.empty_like(r)
        for bits, ids in mask.items():
            if len(ids) == 0:
                continue
            size = {16: 2, 32: 4, 64: 8}[bits]
            self.fun["local" + str(bits)](
                (len(ids),),
                (bs * s,),
                (a[bits], ix, r, out, ids, np.int32(bs), np.int32(s), np.int32(t)),
                shared_mem=2 * bs * s * size,
            )
        return out


class BJAC:
    def __init__(self, kernels, a, ix, bs, t, p, mask, fp64=False):
        self.k = kernels
        self.ix = ix
        self.bs = bs
        self.t = t
        self.p = p
        self.dtype = cp.float64 if fp64 else cp.float32
        self.a = {64: a} if fp64 else {32: a.astype(cp.float32)}
        if not fp64 and np.any(mask):
            self.a[16] = a.astype(cp.float16)
        self.groups = (
            {64: cp.arange(len(mask), dtype=cp.int32)}
            if fp64
            else {
                16: cp.asarray(np.flatnonzero(mask), dtype=cp.int32),
                32: cp.asarray(np.flatnonzero(~mask), dtype=cp.int32),
            }
        )
        self.calls = 0

    def __call__(self, r):
        rhs = cp.ascontiguousarray(r, dtype=self.dtype)
        y = cp.zeros_like(rhs)
        for _ in range(self.p):
            residual = rhs - self.k.sp(self.a[64 if self.dtype == cp.float64 else 32], self.ix, y)
            corr = self.k.local(self.a, self.ix, residual, self.bs, self.t, self.groups)
            y = y + corr
        self.calls += 1
        return y.astype(cp.float64)


class Meter:
    def __init__(self, enabled=False):
        self.enabled = enabled
        self.events = []

    def call(self, name, fn):
        if not self.enabled:
            return fn()
        a, b = cp.cuda.Event(), cp.cuda.Event()
        a.record()
        value = fn()
        b.record()
        self.events.append((name, a, b))
        return value

    def result(self):
        cp.cuda.get_current_stream().synchronize()
        out = {}
        for name, a, b in self.events:
            out[name] = out.get(name, 0) + cp.cuda.get_elapsed_time(a, b) / 1000
        return out


def solve(kern, a, ix, b, pre, tol=1e-8, maxit=400, check_every=5, profile=False):
    meter = Meter(profile)
    x = cp.zeros_like(b)
    r = b.copy()
    history = []
    bn = cp.linalg.norm(b, axis=0)
    if bool(cp.any(bn == 0).get()):
        raise ValueError("zero RHS unsupported by relative stopping")

    def mv(q):
        return meter.call("outer_spmm", lambda: kern.sp(a, ix, q))

    def pc(r):
        return meter.call("bjac", lambda: pre(r))

    def qr(v):
        return meter.call("qr", lambda: cp.linalg.qr(v, mode="reduced")[0])

    def check(step):
        true = b - mv(x)
        ratios = cp.linalg.norm(true, axis=0) / bn
        worst = float(cp.max(ratios).get())
        history.append(
            {
                "iteration": step,
                "max_true_relative_residual": worst,
                "max_recursive_relative_residual": float(
                    cp.max(cp.linalg.norm(r, axis=0) / bn).get()
                ),
                "gap_norm": float(cp.linalg.norm(r - true).get()),
                "true_relative_residual_per_column": cp.asnumpy(ratios).tolist(),
            }
        )
        return worst

    cp.cuda.get_current_stream().synchronize()
    start = time.perf_counter()
    status = "maxit"
    it = 0
    error = None
    worst = float("inf")
    try:
        with cupyx.errstate(linalg="raise"):
            q = qr(pc(r))
            for it in range(1, maxit + 1):
                w = mv(q)
                c, d = meter.call("gram_products", lambda: (q.T @ w, q.T @ r))
                alpha = meter.call("small_solves", lambda: cp.linalg.solve(c, d))
                x, r = meter.call("updates", lambda: (x + q @ alpha, r - w @ alpha))
                if it % check_every == 0 or it == maxit:
                    worst = check(it)
                    if not np.isfinite(worst):
                        status = "nonfinite"
                        break
                    if worst <= tol:
                        status = "converged"
                        break
                z = pc(r)
                f = meter.call("gram_products", lambda: w.T @ z)
                beta = meter.call("small_solves", lambda: -cp.linalg.solve(c, f))
                v = meter.call("updates", lambda: z + q @ beta)
                q = qr(v)
    except Exception as exc:
        status = "failed"
        error = repr(exc)
    # Final confirmation belongs in timing, even after a failure.
    try:
        worst = check(it)
    except Exception as exc:
        error = str(error) + "; residual check: " + repr(exc)
    cp.cuda.get_current_stream().synchronize()
    elapsed = time.perf_counter() - start
    return {
        "status": status,
        "iterations": it,
        "solve_wall_seconds": elapsed,
        "error": error,
        "max_true_relative_residual": worst,
        "history": history,
        "components_gpu_seconds": meter.result(),
        "preconditioner_calls": pre.calls,
    }
