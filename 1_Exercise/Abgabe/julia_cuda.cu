#include <iostream>
#include "lodepng.h"
#include <sys/time.h>
#include <cstddef>
#include <stdio.h>
#include <thrust/complex.h>

#define X_SIZE 2048
#define Y_SIZE 2048
#define ITER 200
#define RANGE 2.0
#define THRESHOLD 10.0

typedef float type;



void checkError( cudaError_t err, const char* msg )
{
	if (err != cudaSuccess) {
		std::cout << cudaGetErrorString(err) << " " << msg << std::endl;
		exit(-1);
	}
}


__global__ void update(unsigned char* picture){
	const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
	const size_t y = blockIdx.y * blockDim.y + threadIdx.y;
	size_t counter = 0;
	double scaling = (double) X_SIZE / 4.0;
	//const size_t x = idx % X_SIZE;
	//const size_t y = (idx-(idx % X_SIZE))/Y_SIZE;
	thrust::complex<type> init (x/scaling - RANGE, y/scaling - RANGE);
	thrust::complex<type> c (-0.8, 0.2);

	if(x < X_SIZE && y < Y_SIZE){
		while( abs(init) < THRESHOLD ){
			init = init * init + c;
			//printf("%f\n", thrust::abs(picture[idx]));
			++counter;
			if(counter >= ITER)
				break;
		}
		picture[4 * X_SIZE * y + 4 * x + 0] = counter % 255;
		picture[4 * X_SIZE * y + 4 * x + 1] = 0;
		picture[4 * X_SIZE * y + 4 * x + 2] = 0;
		picture[4 * X_SIZE * y + 4 * x + 3] = 255;

	}
	//__syncthreads();
}


inline double getSeconds(){
	struct timeval tp;
	gettimeofday(&tp, NULL);
	return ((double) tp.tv_sec + (double) tp.tv_usec * 1e-6 );
}



int main(int argc, const char * argv[]) {

	if(argc != 3){
		std::cout << "wrong amount of arguments" << std::endl;
		exit(-1);
	}
	const int threadsX = std::atoi(argv[1]);
	const int threadsY = std::atoi(argv[2]);
	const dim3 threadsPerBlock(threadsX, threadsY);
	const dim3 numBlocks(X_SIZE/threadsPerBlock.x, Y_SIZE/threadsPerBlock.y);
	double start_time;
	double end_time;
	double kernel_time;
	double mem_time;


	std::vector<unsigned char> picture (X_SIZE*Y_SIZE*4);

	const long long nBytes = X_SIZE*Y_SIZE*sizeof(unsigned char)*4;

	unsigned char* d_picture;
	checkError(cudaMalloc(&d_picture, nBytes), "malloc d_picture");

	start_time = getSeconds();
	update <<< numBlocks, threadsPerBlock >>>(d_picture);
	checkError( cudaPeekAtLastError(), "peek" );//needed to check the kernel for errors
	checkError( cudaDeviceSynchronize(), "synch2" );
	end_time = getSeconds();
	kernel_time = end_time - start_time;
	//std::cout << "kernel time: " << kernel_time  << std::endl;
	
	checkError( cudaDeviceSynchronize(), "synch3" );

	start_time = getSeconds();
	checkError( cudaMemcpy(&picture[0], d_picture, nBytes, cudaMemcpyDeviceToHost), "memcpy d to h" );
	end_time = getSeconds();
	mem_time = end_time - start_time;
	//std::cout << "memcpy d to h: " << mem_time << std::endl;
	std::cout  << mem_time + kernel_time << std::endl;

	checkError( cudaDeviceSynchronize(), "snych4" );

	unsigned error = lodepng::encode("julia.png", picture.data(), X_SIZE, Y_SIZE);
	if(error) std::cout << "error lodepng: " << error << std::endl;


	checkError( cudaFree(d_picture), "free" );
	return 0;
}

