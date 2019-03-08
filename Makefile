SHELL=/bin/bash

ifeq (, $(shell which xlf))

	F90=gfortran
	CXX=g++
	CC=gcc
	CXXFLAGS=-Wall -O2 -g -fno-exceptions -fopenmp
	FFLAGS=-Wall -O2 -g -fopenmp

else

	F90=xlf_r
	CXX=xlc++_r
	CC=xlc_r

	CXXFLAGS=-g -std=c++11 -Wall -qsmp=omp -qoffload -Ofast
	FFLAGS=-g -qsmp=omp -qoffload -Ofast -qsuppress=1501-510 -qextname

endif


all: testomp

testomp: kernelomp.f90 testomp.cpp HybridOMP.H HybridOMP.C Makefile
	$(F90) $(FFLAGS) -c kernelomp.f90
	$(CXX) $(CXXFLAGS) -c HybridOMP.C
	$(CXX) $(CXXFLAGS) testomp.cpp kernelomp.o HybridOMP.o -o $@
	./$@ 100

testacc: kernelacc.f90 testacc.cpp Makefile
	gfortran $(FFLAGS) -fopenacc -c kernelacc.f90
	gcc $(CXXFLAGS)    -fopenacc -std=c++11 testacc.cpp kernelacc.o -o $@
	./$@ 10240000

testcuda: testcuda.cu Makefile
	nvcc -O2 -Wno-deprecated-gpu-targets $< -std=c++11 -g -o $@
	./$@ 10240000


clean:
	rm -f testomp testacc testcuda *.o


benchmark: testomp
	for i in $$(seq 10 29); do \
		./testomp $$((2**$$i)); \
	done

benchmarkcuda: testcuda
	for i in $$(seq 10 30); do \
		./testcuda $$((2**$$i)); \
	done
