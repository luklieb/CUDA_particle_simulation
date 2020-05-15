CXX = nvcc

CXXFLAGS = -O3 -std=c++11 -Wno-deprecated-gpu-targets

all: md_brute md_cell_p md_particle_p md_list

md_brute: md_periodic.cu
		$(CXX) $(CXXFLAGS) -DBRUTE -arch=sm_20 $^ -o $@
md_particle_p: md_periodic.cu
		$(CXX) $(CXXFLAGS) -DPARTICLEP -arch=sm_20 $^ -o $@
md_cell_p: md_periodic.cu
		$(CXX) $(CXXFLAGS) -DCELLP -arch=sm_20 $^ -o $@
md_list: md_periodic.cu
		$(CXX) $(CXXFLAGS) -DLIST -arch=sm_20 $^ -o $@

				
.PHONY: clean
clean: 
		rm -f md_brute md_cell_p md_particle_p md_list *.vtk *.out
