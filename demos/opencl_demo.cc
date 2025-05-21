#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#include <stdio.h>
#include <stdlib.h>
#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#include <CL/cl_ext.h>
#endif
#include <ctime>
#include <time.h>
#include <math.h>

size_t VECTOR_SIZE = pow(2,27);

// OpenCL kernel for each work item
const char *leakyReLUKernel =
        "__kernel                                   \n"
        "void leakyReLUKernel(float leak,          \n"
        "                  __global float *input,   \n"
        "                  __global float *output)  \n"
        "{                                          \n"
        "    //Get index                            \n"
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

int main(void) {
    struct timespec start, finish;
    double elapsed;

    /// Allocate space for ReLU inputs and outputs (vectors)
    float leak = 0.01;
    float *input = (float*)malloc(sizeof(float)*VECTOR_SIZE);
    float *output = (float*)malloc(sizeof(float)*VECTOR_SIZE);
    /// initialize data
    srand (time(NULL) );
    size_t i;
    for(i = 0; i < VECTOR_SIZE; i++)
    {
        input[i] = rand() % 100/100.0-0.5;
        output[i] = 0;
    }

    /// Define platform and queues
    cl_platform_id * platforms = NULL;
    cl_uint     numPlatforms;
    /// Setup platform
    cl_int clStatus = clGetPlatformIDs(0, NULL, &numPlatforms);
    platforms = (cl_platform_id *)malloc(sizeof(cl_platform_id)*numPlatforms);
    clStatus = clGetPlatformIDs(numPlatforms, platforms, NULL);

    cl_device_id* deviceList = NULL;
    cl_uint numDevices;

    /// get num of devices
    clStatus = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, 0,NULL, &numDevices);
    /// allocate memory
    deviceList = (cl_device_id *)malloc(sizeof(cl_device_id)*numDevices);
    clStatus = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, numDevices, deviceList, NULL);

    /// Create context
    cl_context context;
    context = clCreateContext(NULL, numDevices, deviceList, NULL, NULL, &clStatus);

    /// Create command queue
    cl_command_queue commandQueue = clCreateCommandQueue(context, deviceList[0], 0, &clStatus);

    /// Define memory objects
    cl_mem inputClmem = clCreateBuffer(context, CL_MEM_READ_ONLY,VECTOR_SIZE * sizeof(float), NULL, &clStatus);
    cl_mem outputClmem = clCreateBuffer(context, CL_MEM_WRITE_ONLY,VECTOR_SIZE * sizeof(float), NULL, &clStatus);

    /// start time OpenCL
    clock_gettime(CLOCK_MONOTONIC, &start);

    /// copy arguments to the device
    clStatus = clEnqueueWriteBuffer(commandQueue, inputClmem, CL_TRUE, 0, VECTOR_SIZE * sizeof(float),
                                    input, 0, NULL, NULL);
    clStatus = clEnqueueWriteBuffer(commandQueue, outputClmem, CL_TRUE, 0, VECTOR_SIZE * sizeof(float),
                                    output, 0, NULL, NULL);

    /// Create the program
    cl_program program = clCreateProgramWithSource(context, 1,(const char **)&leakyReLUKernel, NULL, &clStatus);

    /// Build program
    clStatus = clBuildProgram(program, 1, deviceList, NULL, NULL, NULL);

    if (clStatus!=CL_SUCCESS){
        printf("Error building program\n");
    }


    /// Create the kernel
    cl_kernel kernel = clCreateKernel(program, "leakyReLUKernel", &clStatus);

    /// Define arguments
    clStatus = clSetKernelArg(kernel, 0, sizeof(float), (void *)&leak);
    clStatus = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&inputClmem);
    clStatus = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&outputClmem);

    /// Execute the kernel
    size_t global_size = VECTOR_SIZE; // Process the entire lists
    size_t local_size = 64;           // Process one item at a time
    clStatus = clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL, &global_size, &local_size, 0, NULL, NULL);

    /// Download results
    clStatus = clEnqueueReadBuffer(commandQueue, outputClmem, CL_TRUE, 0, VECTOR_SIZE * sizeof(float), output,
                                   0, NULL, NULL);

    /// finish
    clStatus = clFlush(commandQueue);
    clStatus = clFinish(commandQueue);

    /// stop time OpenCL
    clock_gettime(CLOCK_MONOTONIC, &finish);
    for(i = 0; i < 10; i++)
    {
        printf("input: %f, output: %f\n", input[i], output[i]);
    }

    /// clean and release memory
    clStatus = clReleaseKernel(kernel);
    clStatus = clReleaseProgram(program);
    clStatus = clReleaseMemObject(inputClmem);
    clStatus = clReleaseMemObject(outputClmem);
    clStatus = clReleaseCommandQueue(commandQueue);
    clStatus = clReleaseContext(context);

    elapsed = (finish.tv_sec - start.tv_sec);
    elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    printf("elapsed time OpenCL: %f\n", elapsed);

    clock_gettime(CLOCK_MONOTONIC, &start);
    for(i = 0; i < VECTOR_SIZE; i++)
    {
        leakyReLU_c(leak, input,output,i);
        if (i<10)
            printf("input: %f, output: %f\n", input[i], output[i]);
    }
    clock_gettime(CLOCK_MONOTONIC, &finish);
    elapsed = (finish.tv_sec - start.tv_sec);
    elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    printf("elapsed time CPU: %f\n", elapsed);

    free(input);
    free(output);
    free(platforms);
    free(deviceList);

    return 0;
}
