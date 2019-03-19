#include <stdio.h>
#include <omp.h>
#include <chrono>
#include <stdlib.h>
#include "HybridOMP.H"
#include "cuda_runtime.h"

using namespace std;

extern "C" void zaxpy_(long *start, long *end, long *len, double *X, double *Y, double *Z);


int main(int argc, char const* argv[])
{
  double *X, *Y, *Z;
  long len = atol(argv[1]);

  cudaMallocManaged(&X, len * sizeof(double));
  cudaMallocManaged(&Y, len * sizeof(double));
  cudaMallocManaged(&Z, len * sizeof(double));

  hyb_num_gpu_available();

  for (long i=0; i<len; i++) {
    X[i] = 1.5;
    Y[i] = 2.3;
    Z[i] = 0.0;
  }

  cudaMemPrefetchAsync(X, len * sizeof(double), 0);
  cudaMemPrefetchAsync(Y, len * sizeof(double), 0);
  cudaMemPrefetchAsync(Z, len * sizeof(double), 0);

  cudaDeviceSynchronize();

  chrono::steady_clock::time_point begin, end;
  constexpr int num_iter = 10;

// #pragma omp target data map(tofrom:Z[:len],Y[:len],X[:len])
{
  long one = 1;

  {
    // Measure time for calculation only
    begin = chrono::steady_clock::now();
    for(int n=0; n<num_iter; n++) {
    #pragma omp target teams is_device_ptr(X, Y, Z) device(0)
      zaxpy_(&one, &len, &len, X, Y, Z);
    }
    end = chrono::steady_clock::now();
  }
}

  for (long i=0; i<len; i++) {
    if (Z[i]!=38.3) {
        printf("Verification failed elem %ld value %lf\n", i, Z[i]);
    }
  }

  printf("%ld %ld microseconds\n", len, chrono::duration_cast<chrono::microseconds>(end - begin).count()/num_iter);

  cudaFree(X);
  cudaFree(Y);
  cudaFree(Z);

  return 0;
}
