#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#include "NNets/nnets_cl.h"
#include <stdio.h>
#include <stdlib.h>
#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#include <CL/cl_ext.h>
#endif

cl_kernel kernel;
cl_mem inputClmem;
cl_mem outputClmem;
cl_command_queue commandQueue;
cl_program program;
cl_context context;

/// Define platform and queues
cl_platform_id * platforms = NULL;
cl_uint     numPlatforms;

/// devices
cl_device_id* deviceList = NULL;
cl_uint numDevices;

// OpenCL kernel for each work item
const char *leakyReLU_kernel =
        "__kernel                                   \n"
        "void leakyReLU_kernel(float leak,          \n"
        "                  __global float *input,   \n"
        "                  __global float *output)  \n"
        "{                                          \n"
        "    //Get the index                        \n"
        "    int index = get_global_id(0);          \n"
        "    if (input[index]>0)                    \n"
        "       output[index]=input[index];         \n"
        "    else                                   \n"
        "       output[index]=leak*input[index];    \n"
        "}                                          \n";

void leakyReLU_c(float leak, float *input, float *output, int index)
{
    if (input[index]>0)
        output[index]=input[index];
    else
        output[index]=leak*input[index];
}

void initializeOpenCL(int vector_size)
{
    //Set up the Platform
    cl_int clStatus = clGetPlatformIDs(0, NULL, &numPlatforms);
    platforms = (cl_platform_id *)malloc(sizeof(cl_platform_id)*numPlatforms);
    clStatus = clGetPlatformIDs(numPlatforms, platforms, NULL);

    /// get num of devices
    clStatus = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, 0,NULL, &numDevices);
    /// allocate memory
    deviceList = (cl_device_id *)malloc(sizeof(cl_device_id)*numDevices);
    clStatus = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, numDevices, deviceList, NULL);

    /// Create context
    context = clCreateContext(NULL, numDevices, deviceList, NULL, NULL, &clStatus);

    /// Create command queue
    commandQueue = clCreateCommandQueue(context, deviceList[0], 0, &clStatus);

    /// Define memory objects
    inputClmem = clCreateBuffer(context, CL_MEM_READ_ONLY, vector_size * sizeof(float), NULL, &clStatus);
    outputClmem = clCreateBuffer(context, CL_MEM_WRITE_ONLY, vector_size * sizeof(float), NULL, &clStatus);

    /// Create the program
    program = clCreateProgramWithSource(context, 1,(const char **)&leakyReLU_kernel, NULL, &clStatus);

    /// Build program
    clStatus = clBuildProgram(program, 1, deviceList, NULL, NULL, NULL);
    if (clStatus!=CL_SUCCESS){
        printf("Error building program\n");
    }

    /// Create the kernel
    kernel = clCreateKernel(program, "leakyReLU_kernel", &clStatus);
}

void computeoutput(float leak, float *input, float *output, int vector_size){
    /// copy arguments to the device
    cl_int clStatus = clEnqueueWriteBuffer(commandQueue, inputClmem, CL_TRUE, 0, vector_size * sizeof(float),
                                    input, 0, NULL, NULL);
    if (clStatus!=0)
    {
        printf("Error %d\n",clStatus);
        if (clStatus == CL_OUT_OF_HOST_MEMORY){
            printf("CL_OUT_OF_HOST_MEMORY\n");
        }
        else if (clStatus == CL_MEM_OBJECT_ALLOCATION_FAILURE){
            printf("CL_MEM_OBJECT_ALLOCATION_FAILURE\n");
        }
        else if (clStatus == CL_INVALID_MEM_OBJECT){
            printf("CL_INVALID_MEM_OBJECT\n");
        }
        return;
    }

    /// Define arguments
    clStatus = clSetKernelArg(kernel, 0, sizeof(float), (void *)&leak);
    clStatus = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&inputClmem);
    clStatus = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&outputClmem);

    /// Execute the kernel
    size_t global_size = vector_size; // Process the entire lists
    size_t local_size = 64;           // Process one item at a time
    clStatus = clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL, &global_size, &local_size, 0, NULL, NULL);

    // multiple layers - uncomment this to see advantages of OpenCL
    // (do the same for the CPU example in the cpp_demo.cpp)
    int layer_no=0;
    clStatus = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&outputClmem);
    for (layer_no=0;layer_no<10;layer_no++){
        clStatus = clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL, &global_size, &local_size, 0, NULL, NULL);
    }

    /// Download results
    clStatus = clEnqueueReadBuffer(commandQueue, outputClmem, CL_TRUE, 0, vector_size * sizeof(float), output,
                                   0, NULL, NULL);

    /// finish
    clStatus = clFlush(commandQueue);
    clStatus = clFinish(commandQueue);
}

void releaseOpenCL(){
    /// clean and release memory
    cl_int clStatus = clReleaseKernel(kernel);
    if (clStatus!=0)
        printf("Error release\n");
    clStatus = clReleaseProgram(program);
    clStatus = clReleaseMemObject(inputClmem);
    clStatus = clReleaseMemObject(outputClmem);
    clStatus = clReleaseCommandQueue(commandQueue);
    clStatus = clReleaseContext(context);

    free(platforms);
    free(deviceList);
}
