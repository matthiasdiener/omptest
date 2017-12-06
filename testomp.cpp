#include <stdio.h>
#include <omp.h>
#include <chrono>
#include <stdlib.h>

using namespace std;

extern "C" void zaxpy_(long *start, long *end, long *len, double *X, double *Y, double *Z);


int main(int argc, char const* argv[])
{
  double *X, *Y, *Z;
  long len = atoi(argv[1]);
  int i;

  X = (double *) malloc(len * sizeof(double));
  Y = (double *) malloc(len * sizeof(double));
  Z = (double *) malloc(len * sizeof(double));

  // warm up offloading
  #pragma omp target
  {

  }

  for (i=0; i<len; i++) {
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

  printf("%ld %.1f %.1f %ld microseconds\n", len, Z[0], Z[len-1], chrono::duration_cast<chrono::microseconds>(end - begin).count());
  return 0;
}
