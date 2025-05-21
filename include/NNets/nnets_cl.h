/** @file nnets_cl.h
 *
 * OpenCL example - leakyReLU
 *
 * @author Dominik Belter
 */

#ifndef H_NNETS
#define H_NNETS

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#include <stdio.h>
#include <stdlib.h>
#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

/// initialize OpenCL
void initializeOpenCL(int vector_size);
/// leakyReLU using CPU
void leakyReLU_c(float leak, float *input, float *output, int index);
/// leakyReLU using OpenCL
void computeoutput(float leak, float *input, float *output, int vector_size);
/// Release OpenCL
void releaseOpenCL();

#ifdef __cplusplus
}
#endif

#endif /* H_NNETS */
