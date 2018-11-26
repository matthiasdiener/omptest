F90=gfortran
CXX=g++
CC=gcc

CXXFLAGS=-Wall -O2 -g -fno-exceptions
FFLAGS=-Wall -O2 -g

all: testomp

testomp: kernelomp.f90 testomp.cpp Makefile
	$(F90) $(FFLAGS) -fopenmp -c kernelomp.f90
	$(CXX) $(CXXFLAGS) -fopenmp testomp.cpp kernelomp.o -o $@
	LD_LIBRARY_PATH=../hybridOMP/gcc-offload/install/lib64/ ./$@ 10240000

testacc: kernelacc.f90 testacc.cpp Makefile
	gfortran $(FFLAGS) -fopenacc -c kernelacc.f90
	gcc $(CXXFLAGS)    -fopenacc -std=c++11 testacc.cpp kernelacc.o -o $@
	./$@ 10240000

testcuda: testcuda.cu Makefile
	/usr/local/cuda-8.0/bin/nvcc -O2 -Wno-deprecated-gpu-targets $< -std=c++11 -g -o $@
	./$@ 10240000


clean:
	rm -f testomp testacc testcuda *.o
