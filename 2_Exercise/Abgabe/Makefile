CXX = nvcc

INCLUDES =
CXXFLAGS = -O3 -std=c++11
LDFLAGS =
LIBS =

target = hesp
src = 
obj = 

#all: juliaCUDA juliaCPU juliaCL

$(target): mdsim.o
		$(CXX) $(CXXFLAGS) -I$(INCLUDES) -arch=sm_20 $^ -o $@

mdsim.o: mdsim.cu
		$(CXX) $(CXXFLAGS) -c $^ -o $@

#parameters.o: parameters.cpp
#		$(CXX) $(CXXFLAGS) -c $^ -o $@ 

				
.PHONY: clean
clean: 
		rm -f parameters.o mdsim.o hesp
