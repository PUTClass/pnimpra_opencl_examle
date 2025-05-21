/** @file cpp_demo.cpp
 *
 * OpenCL example - ReLU example
 *
 * @author Dominik Belter
 */

#include <iostream>
#include <vector>
#include <random>
#include <chrono>

extern "C" {
#include "NNets/nnets_cl.h"
}

int main()
{
    try {
        size_t vectorSize = pow(2,7);
        std::vector<float> input(vectorSize,0.0);
        std::vector<float> output(vectorSize,0.0);
        /// initialize data
        srand (time(NULL) );
        std::default_random_engine generator;
        std::uniform_real_distribution<float> distribution(-1.0,1.0);
        for(size_t i = 0; i < vectorSize; i++)
        {
            input[i] = distribution(generator);
            output[i] = 0;
        }

        initializeOpenCL(vectorSize);
        float leak=0.01;
        //compute output for all neurons using OpenCL
        std::chrono::steady_clock::time_point beginCL = std::chrono::steady_clock::now();
        computeoutput(leak,&input.front(),&output.front(),vectorSize);
        std::chrono::steady_clock::time_point endCL = std::chrono::steady_clock::now();
        std::cout << "Time difference for OpenCL= " <<
                     std::chrono::duration_cast<std::chrono::microseconds>(endCL - beginCL).count() << "[µs]\n";
        for(size_t i = 0; i<10; i++)
        {
            std::cout << "input:736 " << input[i] << ", output: " << output[i] <<"\n";
        }

        //compute output for all neurons using CPU
        std::chrono::steady_clock::time_point beginCPU = std::chrono::steady_clock::now();
        for(size_t i = 0; i < vectorSize; i++)
        {
            leakyReLU_c(leak, &input.front(), &output.front(), i);
            // multiple layers
            for (size_t layerNo=0;layerNo<10;layerNo++)
                leakyReLU_c(leak, &output.front(), &output.front(), i);
        }
        std::chrono::steady_clock::time_point endCPU = std::chrono::steady_clock::now();
        std::cout << "Time difference for CPU= " <<
                     std::chrono::duration_cast<std::chrono::microseconds>(endCPU - beginCPU).count() << "[µs]\n";
        for(size_t i = 0; i<10; i++)
        {
            std::cout << "input: " << input[i] << ", output: " << output[i] <<"\n";
        }
    }
    catch (const std::exception& ex) {
        std::cerr << ex.what() << std::endl;
        return 1;
    }
    return 0;
}
