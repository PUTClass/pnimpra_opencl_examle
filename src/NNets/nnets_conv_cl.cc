#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#include "NNets/nnets_conv_cl.h"
#include <stdio.h>
#include <stdlib.h>
#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#define MASK_SIZE 3

cl_kernel kernelConv;
cl_mem inputConvClmem;
cl_mem outputConvClmem;
cl_mem maskConvClmem;
cl_command_queue commandQueueConv;
cl_program programConv;
cl_context contextConv;

/// Define platform and queues
cl_platform_id * platformsConv = NULL;
cl_uint     numplatformsConv;

/// devices
cl_device_id* deviceListConv = NULL;
cl_uint numDevicesConv;

const char *convKernel =
"__kernel                                   \n"
"void convKernel(__global uchar *input,   \n"
"                  __global char *mask,  \n"
"                  __global uchar *output,  \n"
"                  const int rows,  \n"
"                  const int cols)  \n"
"{                                          \n"
"// Write you custom convolutional kernel here. Make sure you use proper data types for numerical calculations. \n"
"// The function takes in several inputs, which are needed to convert between 2D and 1D indices. \n"
"// For now the function copies the input image into the output vector. \n"
"   const int index=get_global_id(0); \n"
"   output[index] = (uchar)input[index];\n"
"}\n";

void check(cl_int err, const char* msg) {
    if (err != CL_SUCCESS) {
        fprintf(stderr, "OpenCL Error (%d): %s\n", err, msg);
        exit(1);
    }
}

void print_build_info(cl_int err) {
    if(err!=0){
    size_t log_size;
    clGetProgramBuildInfo(programConv, deviceListConv[0], CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);

    // Allocate space for the log
    char* log = (char*)malloc(log_size);

    // Get the log
    clGetProgramBuildInfo(programConv, deviceListConv[0], CL_PROGRAM_BUILD_LOG, log_size, log, NULL);

    // Print the log
    printf("Build Log:\n%s\n", log);

    // Cleanup
    free(log);
    exit(0);
    }
}

/// convolution using CPU
void conv_c(unsigned char *input, char *mask, unsigned char *output, int rows, int cols, int index)
{
    int rowNo = index/cols;
    int colNo = index%cols;
    if (rowNo>0&&rowNo<rows-1&&colNo>0&&colNo<cols){
        int row,col;
        int sum=0;
        for (row=-1;row<2;row++){
            for (col=-1;col<2;col++){
                int maskIndex = (row+1)*3+(col+1);
                int imageIndex = (rowNo+row)*cols+(colNo+col);
                sum+= input[imageIndex]*mask[maskIndex];
            }
        }
        output[index] = (unsigned char)abs(sum);
    }
}

void initializeOpenCLConv(int rows, int cols)
{
    int vector_size = rows*cols;
    
    //Set up the Platform
    cl_int clStatus = clGetPlatformIDs(0, NULL, &numplatformsConv);
    platformsConv = (cl_platform_id *)malloc(sizeof(cl_platform_id)*numplatformsConv);
    clStatus = clGetPlatformIDs(numplatformsConv, platformsConv, NULL);
    /// get num of devices
    clStatus = clGetDeviceIDs(platformsConv[0], CL_DEVICE_TYPE_GPU, 0,NULL, &numDevicesConv);
    /// allocate memory
    deviceListConv = (cl_device_id *)malloc(sizeof(cl_device_id)*numDevicesConv);
    clStatus = clGetDeviceIDs(platformsConv[0], CL_DEVICE_TYPE_GPU, numDevicesConv, deviceListConv, NULL);
    /// Create context
    contextConv = clCreateContext(NULL, numDevicesConv, deviceListConv, NULL, NULL, &clStatus);
    /// Create command queue
    commandQueueConv = clCreateCommandQueue(contextConv, deviceListConv[0], 0, &clStatus);
    /// Define memory objects
    inputConvClmem = clCreateBuffer(contextConv, CL_MEM_READ_ONLY, vector_size * sizeof(unsigned char), NULL, &clStatus);
    maskConvClmem = clCreateBuffer(contextConv, CL_MEM_READ_ONLY, MASK_SIZE*MASK_SIZE * sizeof(char), NULL, &clStatus);
    outputConvClmem = clCreateBuffer(contextConv, CL_MEM_READ_WRITE, vector_size * sizeof(unsigned char), NULL, &clStatus);
    /// Create the program
    programConv = clCreateProgramWithSource(contextConv, 1,(const char **)&convKernel, NULL, &clStatus);
    /// Build program
    clStatus = clBuildProgram(programConv, 1, deviceListConv, NULL, NULL, NULL);
    
    print_build_info(clStatus);
    
    /// Create the kernel
    kernelConv = clCreateKernel(programConv, "convKernel", &clStatus);
}

void computeoutputConv(unsigned char *input, char *mask, unsigned char *output, int rows, int cols){
    /// copy arguments to the device
    cl_int clStatus = clEnqueueWriteBuffer(commandQueueConv, inputConvClmem, CL_TRUE, 0, rows*cols * sizeof(unsigned char),
                                            input, 0, NULL, NULL);
    clStatus = clEnqueueWriteBuffer(commandQueueConv, maskConvClmem, CL_TRUE, 0, MASK_SIZE*MASK_SIZE * sizeof(char),
                                            mask, 0, NULL, NULL);
    clStatus = clEnqueueWriteBuffer(commandQueueConv, outputConvClmem, CL_TRUE, 0, rows*cols * sizeof(unsigned char),
                                           output, 0, NULL, NULL);

    clStatus = clSetKernelArg(kernelConv, 0, sizeof(cl_mem), &inputConvClmem);
    clStatus = clSetKernelArg(kernelConv, 1, sizeof(cl_mem), &maskConvClmem);
    clStatus = clSetKernelArg(kernelConv, 2, sizeof(cl_mem), &outputConvClmem);
    clStatus = clSetKernelArg(kernelConv, 3, sizeof(int), &rows);
    clStatus = clSetKernelArg(kernelConv, 4, sizeof(int), &cols);
    
    /// Execute the kernel
    size_t global_size = cols*rows; // Process the entire lists
    size_t local_size = 32;           // Process one item at a time. It might lead to runtime error if set too high.
    clStatus = clEnqueueNDRangeKernel(commandQueueConv, kernelConv, 1, NULL, &global_size, &local_size,
                                      0, NULL, NULL);
    check(clStatus, "Kernel operations");
    /// Download results

    clStatus = clEnqueueReadBuffer(commandQueueConv, outputConvClmem, CL_TRUE, 0, cols*rows * sizeof(unsigned char),
                                   output, 0, NULL, NULL);
    check(clStatus, "Reading data");

    /// finish
    clStatus = clFlush(commandQueueConv);
    clStatus = clFinish(commandQueueConv);
}

void releaseOpenCL(){
    /// clean and release memory
    cl_int clStatus = clReleaseKernel(kernelConv);
    if (clStatus!=0)
        printf("Error release\n");
    clStatus = clReleaseProgram(programConv);
    clStatus = clReleaseMemObject(inputConvClmem);
    clStatus = clReleaseMemObject(outputConvClmem);
    clStatus = clReleaseMemObject(maskConvClmem);
    clStatus = clReleaseCommandQueue(commandQueueConv);
    clStatus = clReleaseContext(contextConv);

    free(platformsConv);
    free(deviceListConv);
}
