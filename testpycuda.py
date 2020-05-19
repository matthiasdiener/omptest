#!/usr/bin/env python3

import pycuda.driver as cuda
from pycuda.autoinit import context
from pycuda.compiler import SourceModule

import numpy as np
import time
import sys

size = int(sys.argv[1])

threadsPerBlock = 256
blocksPerGrid = 1

a = np.full(size,1.5).astype(np.float64)
b = np.full(size,2.3).astype(np.float64)
res = np.random.rand(size).astype(np.float64)

a_gpu = cuda.mem_alloc(a.nbytes)
b_gpu = cuda.mem_alloc(b.nbytes)
res_gpu = cuda.mem_alloc(res.nbytes)

cuda.memcpy_htod(a_gpu, a)
cuda.memcpy_htod(b_gpu, b)
cuda.memcpy_htod(res_gpu, res)

mod = SourceModule("""
__global__ void zaxpy( double *a,  double *b, double *res, long numElements)
{
    long i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements)
    {
        res[i] = 23.0 * a[i] + b[i];
    }
}
""")

func = mod.get_function("zaxpy")


# Warmup
func(a_gpu, b_gpu, res_gpu, np.int64(size), block=(16,16,1))
func(a_gpu, b_gpu, res_gpu, np.int64(size), block=(16,16,1))
func(a_gpu, b_gpu, res_gpu, np.int64(size), block=(16,16,1))

context.synchronize() 

print(time.time())
start = time.time()
for i in range(10):
    func(a_gpu, b_gpu, res_gpu, np.int64(size), block=(16,16,1), grid=(16,16,1))
    context.synchronize() 
end = time.time()

print(time.time())

print(size, (end - start)* 1000*1000/10) #microseconds
