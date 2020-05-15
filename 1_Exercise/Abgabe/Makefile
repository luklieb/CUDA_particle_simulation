CXX =

INCLUDES = /usr/include/thrust
CXXFLAGS = -O3 -std=c++11
LDFLAGS = -lOpenCL
LIBS =

target = 
src = 
obj = 

all: juliaCUDA juliaCPU juliaCL cudaQuery openclQuery

juliaCUDA: julia_cuda.cu lodepng.cpp
		nvcc $(CXXFLAGS) -I$(INCLUDES) -arch=sm_20 $^ -o $@

juliaCPU: julia_cpu.cpp lodepng.cpp
		g++ $(CXXFLAGS) $^ -o $@

juliaCL: juliaCL.cpp lodepng.cpp
		g++ $(CXXFLAGS) $^  -o $@ $(LDFLAGS)

cudaQuery: cuda_query.cu
		nvcc $(CXXFLAGS) -I$(INCLUDES) -arch=sm_20 $^ -o $@

openclQuery: opencl_query.c
		gcc -std=c99  -o $@ $^  $(LDFLAGS)
				
.PHONY: clean
clean: 
		rm -f juliaCUDA juliaCPU juliaCL cudaQuery openclQuery
