#include <iostream>
#include <cuda_runtime.h>

using std::cout;
using std::endl;

int main(){




	int numDev;
	cudaGetDeviceCount( &numDev );

	for(int i = 0; i<numDev; ++i){

		cudaDeviceProp properties;
		cudaGetDeviceProperties(&properties, i);
		cout << "Device #: " << i << endl;
		cout << "Name: " << properties.name << endl;
		cout << "Global Mem: " << properties.totalGlobalMem << endl;
		cout << "sharedMemPerBlock: " << properties.sharedMemPerBlock << endl;
		cout << "warpSize: " << properties.warpSize << endl;
		cout << "maxThreadsPerBlock: " << properties.maxThreadsPerBlock << endl;
		cout << "maxThreads X: " << properties.maxThreadsDim[0] << endl;
		cout << "maxThreads Y: " << properties.maxThreadsDim[1] << endl;
		cout << "maxThreads Z: " << properties.maxThreadsDim[2] << endl;
		cout << "maxGridSize X: " << properties.maxGridSize[0] << endl;
		cout << "maxGridSize Y: " << properties.maxGridSize[1] << endl;
		cout << "maxGridSize Z: " << properties.maxGridSize[2] << endl;
		cout << "clockrate: " << properties.clockRate << endl;
		cout << "constMem: " << properties.totalConstMem << endl;


	}	



}
