//frei nach: https://www.fixstars.com/en/opencl/book/OpenCLProgrammingBook/first-opencl-program/

#include <stdio.h>
#include <stdlib.h>
#include <CL/opencl.h>


#define MEM_SIZE (128)
#define MAX_NAME_LEN 1000

void cl_error_to_str(cl_int e);


int main (){

	cl_device_id device = NULL;
	cl_platform_id platform = NULL;
	cl_uint ret_num_devices;
	cl_uint ret_num_platforms;
	cl_ulong mem_size;
	cl_uint cache_size;
	size_t group_size;
	char buf[MAX_NAME_LEN];
	

	cl_error_to_str( clGetPlatformIDs( 1, &platform, &ret_num_platforms) );
	cl_error_to_str( clGetDeviceIDs( platform, CL_DEVICE_TYPE_DEFAULT, 1, &device, &ret_num_devices) );



	// get platform vendor name
	cl_error_to_str( clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, sizeof(buf), buf, NULL) );
	printf("platform vendor: '%s'\n",  buf);
	cl_error_to_str( clGetPlatformInfo(platform, CL_PLATFORM_NAME, sizeof(buf), buf, NULL) );
	printf("platform name: '%s'\n", buf);
	cl_error_to_str( clGetPlatformInfo(platform, CL_PLATFORM_VERSION, sizeof(buf), buf, NULL) );
	printf("platform version:  '%s'\n", buf);
	cl_error_to_str( clGetPlatformInfo(platform, CL_PLATFORM_EXTENSIONS, sizeof(buf), buf, NULL) );
	printf("platform extensions: '%s'\n",  buf);

	// get devices in platform
	cl_error_to_str( clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(buf), buf, NULL) );
	printf("device name: '%s'\n", buf);
	cl_error_to_str( clGetDeviceInfo(device, CL_DEVICE_EXTENSIONS, sizeof(buf), buf, NULL) );
	printf("device extensions: '%s'\n", buf);
	cl_error_to_str( clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, sizeof(cl_ulong), &mem_size, NULL) );
	printf("device global mem cache size: '%i'\n", mem_size);
	cl_error_to_str( clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(buf), buf, NULL) );
	printf("device name: '%s'\n", buf);
	cl_error_to_str( clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &cache_size, NULL) );
	printf("device max compute unites: '%i'\n", cache_size);
	cl_error_to_str( clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &group_size, NULL) );
	printf("device max work group size: '%i'\n", group_size);
	cl_error_to_str( clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(cl_uint), &cache_size, NULL) );
	printf("device max work item dimensions: '%i'\n", cache_size);
	cl_error_to_str( clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_ulong), &mem_size, NULL) );
	printf("device local mem size: '%i'\n", mem_size);
	



	








}





void cl_error_to_str(cl_int e)
{
	switch (e)
	{
		case CL_SUCCESS: break;
		case CL_DEVICE_NOT_FOUND: printf("device not found\n"); break;
		case CL_DEVICE_NOT_AVAILABLE: printf("device not available\n"); break;
		case CL_MEM_OBJECT_ALLOCATION_FAILURE: printf("mem object allocation failure\n"); break;
		case CL_OUT_OF_RESOURCES: printf("out of resources\n"); break;
		case CL_OUT_OF_HOST_MEMORY: printf("out of host memory\n"); break;
		case CL_PROFILING_INFO_NOT_AVAILABLE: printf("profiling info not available\n"); break;
		case CL_MEM_COPY_OVERLAP: printf("mem copy overlap\n"); break;
		case CL_IMAGE_FORMAT_MISMATCH: printf("image format mismatch\n"); break;
		case CL_IMAGE_FORMAT_NOT_SUPPORTED: printf("image format not supported\n"); break;
		case CL_BUILD_PROGRAM_FAILURE: printf("build program failure\n"); break;
		case CL_MAP_FAILURE: printf("map failure\n"); break;
		case CL_INVALID_VALUE: printf("invalid value\n"); break;
		case CL_INVALID_DEVICE_TYPE: printf("invalid device type\n"); break;
		case CL_INVALID_PLATFORM: printf("invalid platform\n"); break;
		case CL_INVALID_DEVICE: printf("invalid device\n"); break;
		case CL_INVALID_CONTEXT: printf("invalid context\n"); break;
		case CL_INVALID_QUEUE_PROPERTIES: printf("invalid queue properties\n"); break;
		case CL_INVALID_COMMAND_QUEUE: printf("invalid command queue\n"); break;
		case CL_INVALID_HOST_PTR: printf("invalid host ptr\n"); break;
		case CL_INVALID_MEM_OBJECT: printf("invalid mem object\n"); break;
		case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR: printf("invalid image format descriptor\n"); break;
		case CL_INVALID_IMAGE_SIZE: printf("invalid image size\n"); break;
		case CL_INVALID_SAMPLER: printf("invalid sampler\n"); break;
		case CL_INVALID_BINARY: printf("invalid binary\n"); break;
		case CL_INVALID_BUILD_OPTIONS: printf("invalid build options\n"); break;
		case CL_INVALID_PROGRAM: printf("invalid program\n"); break;
		case CL_INVALID_PROGRAM_EXECUTABLE: printf("invalid program executable\n"); break;
		case CL_INVALID_KERNEL_NAME: printf("invalid kernel name\n"); break;
		case CL_INVALID_KERNEL_DEFINITION: printf("invalid kernel definition\n"); break;
		case CL_INVALID_KERNEL: printf("invalid kernel\n"); break;
		case CL_INVALID_ARG_INDEX: printf("invalid arg index\n"); break;
		case CL_INVALID_ARG_VALUE: printf("invalid arg value\n"); break;
		case CL_INVALID_ARG_SIZE: printf("invalid arg size\n"); break;
		case CL_INVALID_KERNEL_ARGS: printf("invalid kernel args\n"); break;
		case CL_INVALID_WORK_DIMENSION: printf("invalid work dimension\n"); break;
		case CL_INVALID_WORK_GROUP_SIZE: printf("invalid work group size\n"); break;
		case CL_INVALID_WORK_ITEM_SIZE: printf("invalid work item size\n"); break;
		case CL_INVALID_GLOBAL_OFFSET: printf("invalid global offset\n"); break;
		case CL_INVALID_EVENT_WAIT_LIST: printf("invalid event wait list\n"); break;
		case CL_INVALID_EVENT: printf("invalid event\n"); break;
		case CL_INVALID_OPERATION: printf("invalid operation\n"); break;
		case CL_INVALID_GL_OBJECT: printf("invalid gl object\n"); break;
		case CL_INVALID_BUFFER_SIZE: printf("invalid buffer size\n"); break;
		case CL_INVALID_MIP_LEVEL: printf("invalid mip level\n"); break;

		default: printf("invalid/unknown error code\n"); break;
	}
}
