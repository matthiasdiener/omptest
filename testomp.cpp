#include <stdio.h>
#include <omp.h>
#include <chrono>
#include <stdlib.h>
#include "HybridOMP.H"

using namespace std;

extern "C" void zaxpy_(long *start, long *end, long *len, double *X, double *Y, double *Z);


int main(int argc, char const* argv[])
{
  double *X, *Y, *Z;
  long len = atoi(argv[1]);

  X = (double *) malloc(len * sizeof(double));
  Y = (double *) malloc(len * sizeof(double));
  Z = (double *) malloc(len * sizeof(double));

  hyb_num_gpu_available();

  // warm up offloading
  #pragma omp target
  {

  }

  for (long i=0; i<len; i++) {
    X[i] = 1.5;
    Y[i] = 2.3;
    Z[i] = 0.0;
  }

  chrono::steady_clock::time_point begin, end;

#pragma omp target data map(tofrom:Z[:len],Y[:len],X[:len])
{
    long one = 1;
    
    {
      // Measure time for calculation only
      begin = chrono::steady_clock::now();
      #pragma omp target teams
        zaxpy_(&one, &len, &len, X, Y, Z);
      end = chrono::steady_clock::now();
    }
}

  for (long i=0; i<len; i++) {
    if (Z[i]!=38.3) {
        printf("Verification failed elem %ld value %lf\n", i, Z[i]);
    }
  }

  printf("%ld %ld microseconds\n", len, chrono::duration_cast<chrono::microseconds>(end - begin).count());
  return 0;
}
