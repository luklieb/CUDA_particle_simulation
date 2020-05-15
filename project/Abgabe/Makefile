CXX = nvcc

CXXFLAGS =  -O3 -std=c++11 -Wno-deprecated-gpu-targets -arch=sm_20

all: md_gpu md_cpu md_hybrid
 
md_hybrid: md_hybrid.o gpu.o cpu.o helper.o reader_writer.o
		$(CXX) $(CXXFLAGS) -lgomp $^ -o $@

md_gpu: md_gpu.o gpu.o helper.o reader_writer.o
		$(CXX) $(CXXFLAGS) $^ -o $@

md_cpu: md_cpu.o cpu.o helper.o reader_writer.o
		$(CXX) $(CXXFLAGS) -lgomp $^ -o $@

md_hybrid.o: md_hybrid.cu
		$(CXX) -c $(CXXFLAGS) -dc $^ -o $@

md_cpu.o: md.cu
		$(CXX) -c $(CXXFLAGS) -DCPU -dc -Xcompiler -fopenmp $^ -o $@

md_gpu.o: md.cu
		$(CXX) -c $(CXXFLAGS) -DGPU -dc $^ -o $@

gpu.o: gpu.cu
		$(CXX) -c $(CXXFLAGS) -dc $^ -o $@

cpu.o: cpu.cu
		$(CXX) -c $(CXXFLAGS) -dc -Xcompiler -fopenmp -Xcompiler -Wall  $^ -o $@

helper.o: helper.cu
		$(CXX) -c $(CXXFLAGS) -dc  -Xcompiler -Wall $^ -o $@

reader_writer.o: reader_writer.cpp
		$(CXX) -c $(CXXFLAGS) -dc -Xcompiler -Wall  $^ -o $@

.PHONY: clean
clean: 
		rm -f md_hybrid md_gpu md_cpu *.vtk *.out *.o
