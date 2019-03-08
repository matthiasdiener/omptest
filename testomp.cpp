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

  for (long i=0; i<len; i++) {
    X[i] = 1.5;
    Y[i] = 2.3;
    Z[i] = 0.0;
  }

  chrono::steady_clock::time_point begin, end;
  constexpr int num_iter = 10;
  long leng = len*0.8;
  long lenc = leng+1;
  long one = 1;

#pragma omp parallel num_threads(2)
{

if (omp_get_thread_num() == 0) {
#pragma omp target data map(tofrom:Z[:leng],Y[:leng],X[:leng])
{
  // long s = 1;
  // long e = 100;

  {
    // Measure time for calculation only
    begin = chrono::steady_clock::now();
    for(int n=0; n<num_iter; n++) {
    #pragma omp target teams
      zaxpy_(&one, &leng, &leng, X, Y, Z);
    }
    end = chrono::steady_clock::now();
  }
}
}
else {
	//leng--;
	// Z[leng] =47;
  // lenc++;

  zaxpy_(&leng, &len, &len, X, Y, Z);
}

}

  for (long i=0; i<len; i++) {
    if (Z[i]!=38.3) {
        // printf("%ld %ld Verification failed elem %ld value %lf\n",leng, lenc, i, Z[i]);
	// exit(1);
    }
  }

  printf("%ld %ld microseconds\n", len, chrono::duration_cast<chrono::microseconds>(end - begin).count()/num_iter);
  return 0;
}
