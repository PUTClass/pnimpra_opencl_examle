/** @file nnets_conv_cl.h
 *
 * OpenCL example - convolution
 *
 * @author Dominik Belter
 */

#ifndef H_NNETS_CONV
#define H_NNETS_CONV

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
void initializeOpenCLConv(int rows, int cols);
/// convolution using CPU
void conv_c(unsigned char *input, char *mask, unsigned char *output, int rows, int cols, int index);
/// convolution using OpenCL
void computeoutputConv(unsigned char *input, char *mask, unsigned char *output, int rows, int cols);
/// Release OpenCL
void releaseOpenCL();

#ifdef __cplusplus
}
#endif

#endif /* H_NNETS_CONV */
