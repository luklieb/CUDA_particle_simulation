#define __CL_ENABLE_EXCEPTIONS
#define CL_USE_DEPRECATED_OPENCL_1_1_APIS
#include <CL/cl.hpp>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <utility>
#include <assert.h>
#include <sys/time.h>

#include "lodepng.h"

double getSeconds(){
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double) tp.tv_sec + (double) tp.tv_usec * 1e-6 );
}

int main(int argc , char** argv){
    if (argc != 3) {
        std::cerr << "Error: wrong number of arguments" << std::endl;
        exit(EXIT_FAILURE);
    }

    std::vector<cl::Platform> platforms;
    std::vector<cl::Device> devices;
    cl::Platform::get(&platforms);
   // std::cout << platforms[0].getInfo<CL_PLATFORM_VENDOR>() << std::endl;
    //std::cout << platforms[0].getInfo<CL_PLATFORM_NAME>() << std::endl;
    platforms[0].getDevices(CL_DEVICE_TYPE_ALL, &devices);
    //std::cout << "device name: " << devices[0].getInfo<CL_DEVICE_NAME>() << std::endl;
    //std::cout << "max. work-group: " << devices[0].getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>() << std::endl;
    //std::cout << "max. work-items x: " << devices[0].getInfo<CL_DEVICE_MAX_WORK_ITEM_SIZES>()[0] << std::endl;
    //std::cout << "max. work-items y: " << devices[0].getInfo<CL_DEVICE_MAX_WORK_ITEM_SIZES>()[1] << std::endl;
    //std::cout << "max. compute units: " << devices[0].getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>() << std::endl;

    unsigned int threads_x = std::stoi(argv[1]);
    unsigned int threads_y = std::stoi(argv[2]);
    if (threads_x > devices[0].getInfo<CL_DEVICE_MAX_WORK_ITEM_SIZES>()[0]) {
        std::cerr << "Error: number of threads in x direction exceeds maximum" << std::endl;
        exit(EXIT_FAILURE);
    }
    if (threads_y > devices[0].getInfo<CL_DEVICE_MAX_WORK_ITEM_SIZES>()[1]) {
        std::cerr << "Error: number of threads in y direction exceeds maximum" << std::endl;
        exit(EXIT_FAILURE);
    }
    if (threads_x * threads_y > devices[0].getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>()) {
        std::cerr << "Error: number of threads exceed work group maximum" << std::endl;
        exit(EXIT_FAILURE);
    }


    try {
        cl::Context context(devices[0]);
        //std::ifstream kernel_stream("kernel_double.c");
        std::ifstream kernel_stream("kernel.c");
        std::istreambuf_iterator<char> begin(kernel_stream), end;
        std::string kernel_string(begin, end);
        //std::cout << kernel_string << std::endl;
        cl::Program::Sources sources;
        sources.push_back(std::make_pair(kernel_string.c_str(), kernel_string.length()+1));
        cl::Program program(context, sources);

        program.build(devices);

        cl::CommandQueue cmdqueue(context, devices[0]);
        unsigned int height = 2048, width = 2048;
        unsigned int picsize = width * height * 4;

        // aquire host memory
        cl_uchar* picture_host = new cl_uchar[picsize];
        cl::Buffer picture_device(context, CL_MEM_READ_WRITE, sizeof(cl_uchar) * picsize);
        cl::Kernel kernel(program, "colorJulia");
        // set kernel arguments
        kernel.setArg(0, picture_device);
        kernel.setArg(1, width);
        kernel.setArg(2, height);

        // enqueue kernel to command queue and measure execution time
        double start = getSeconds();
        cmdqueue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(width, height), cl::NullRange);
        cmdqueue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange((width + threads_x - 1) / threads_x * threads_x, (height + threads_y - 1) / threads_y * threads_y), cl::NDRange(threads_x, threads_y));
        cmdqueue.finish();
        cmdqueue.enqueueReadBuffer(picture_device, CL_TRUE, 0, sizeof(cl_uchar) * picsize, picture_host);
 
		double stop = getSeconds();
        std::cout << stop - start << std::endl;

       
        unsigned error = lodepng::encode("julia.png", picture_host, width, height);
        if(error) std::cout << "encoder error " << error << ": "<< lodepng_error_text(error) << std::endl;
        // free host memory
        delete[] picture_host;

    } catch (cl::Error err) {
        std::cerr << "ERROR: " << err.what() << " (" << err.err() << ")" << std::endl;
    }
}
