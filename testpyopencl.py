#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function
import numpy as np
import pyopencl as cl
import time
import sys

size = int(sys.argv[1])

# a_np = np.random.rand(size).astype(np.float64)
# b_np = np.random.rand(size).astype(np.float64)


a_np = np.full(size,1.5).astype(np.float64)
b_np = np.full(size,2.3).astype(np.float64)

# a_np = 1.5

# print(a_np[12])

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

mf = cl.mem_flags
a_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a_np)
b_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b_np)

start = time.time()
cl.enqueue_copy(queue, a_g, a_np)
cl.enqueue_copy(queue, b_g, b_np)



prg = cl.Program(ctx, """
__kernel void sum(
    __global const double *a_g, __global const double *b_g, __global double *res_g)
{
  int gid = get_global_id(0);
  res_g[gid] = 23.0 * a_g[gid] + b_g[gid];
}
""").build()

res_g = cl.Buffer(ctx, mf.WRITE_ONLY, a_np.nbytes)
cl.enqueue_copy(queue, b_g, b_np)
cl.enqueue_barrier(queue)
end = time.time()
# print(size, (end - start)* 1000*1000) #microseconds

start = time.time()
prg.sum(queue, a_np.shape, None, a_g, b_g, res_g)
prg.sum(queue, a_np.shape, None, a_g, b_g, res_g)
prg.sum(queue, a_np.shape, None, a_g, b_g, res_g)
prg.sum(queue, a_np.shape, None, a_g, b_g, res_g)
prg.sum(queue, a_np.shape, None, a_g, b_g, res_g)
prg.sum(queue, a_np.shape, None, a_g, b_g, res_g)
prg.sum(queue, a_np.shape, None, a_g, b_g, res_g)
prg.sum(queue, a_np.shape, None, a_g, b_g, res_g)
# prg.sum(queue, a_np.shape, None, a_g, b_g, res_g)
cl.enqueue_barrier(queue)

end = time.time()
print(size, (end - start)* 1000*1000) #microseconds


# res_np = np.empty_like(a_np)
# cl.enqueue_copy(queue, res_np, res_g)

# print(res_np[12])

# Check on CPU with Numpy:
# print(res_np - (a_np + b_np))
# print(np.linalg.norm(res_np - (a_np + b_np)))
# assert np.allclose(res_np, a_np + b_np)
