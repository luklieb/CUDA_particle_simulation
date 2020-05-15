CXX = nvcc

CXXFLAGS = -O3 -std=c++11 -Wno-deprecated-gpu-targets

all: dem
 
dem: dem.cu
		$(CXX) $(CXXFLAGS) -arch=sm_20 $^ -o $@
				
.PHONY: clean
clean: 
		rm -f dem *.vtk *.out
