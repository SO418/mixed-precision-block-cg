"""CUDA kernels that round every operation to the target format; vectors are stored in FP64.

This implementation is instrumented for accuracy studies and is not tuned for speed.
"""

import numpy as np
from numerics import Backend

CUDA = r"""
#include <cuda_fp16.h>
#include <cuda_bf16.h>
extern "C" __global__ void primitive(const double* a,const double* b,double* y,
                                        long long n,int fmt,int op) {
 long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
 if(i>=n)return;
 if(fmt==16){
  __half x=__double2half(a[i]),z=__double2half(b[i]);
  __half v=op==0?__hadd(x,z):(op==1?__hsub(x,z):__hmul(x,z));
  y[i]=(double)__half2float(v);
 }else if(fmt==17){
  __nv_bfloat16 x=__float2bfloat16_rn((float)a[i]);
  __nv_bfloat16 z=__float2bfloat16_rn((float)b[i]);
  __nv_bfloat16 v=op==0?__hadd(x,z):(op==1?__hsub(x,z):__hmul(x,z));
  y[i]=(double)__bfloat162float(v);
 }else if(fmt==32){
  float x=(float)a[i],z=(float)b[i];
  y[i]=(double)(op==0?__fadd_rn(x,z):(op==1?__fsub_rn(x,z):__fmul_rn(x,z)));
 }else{
  double x=a[i],z=b[i];
  y[i]=op==0?__dadd_rn(x,z):(op==1?__dsub_rn(x,z):__dmul_rn(x,z));
 }
}
"""


class NativeBackend(Backend):
    def __init__(self):
        super().__init__("cuda")
        cp = self.xp
        props = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
        if props["major"] < 8:
            raise RuntimeError("This four-format suite requires Ampere or newer for BF16.")
        self.module = cp.RawModule(
            code=CUDA, options=("--std=c++14", "--fmad=false", "--ftz=false")
        )
        self.kernel = self.module.get_function("primitive")

    def op(self, a, b, kind, fmt):
        cp = self.xp
        a, b = cp.broadcast_arrays(a, b)
        a = cp.ascontiguousarray(a, dtype=cp.float64)
        b = cp.ascontiguousarray(b, dtype=cp.float64)
        y = cp.empty_like(a)
        self.kernel(
            ((a.size + 255) // 256,),
            (256,),
            (
                a,
                b,
                y,
                np.int64(a.size),
                np.int32({"fp16": 16, "bf16": 17, "fp32": 32, "fp64": 64}[fmt]),
                np.int32({"add": 0, "sub": 1, "mul": 2}[kind]),
            ),
        )
        return y, self.env(y, fmt)
