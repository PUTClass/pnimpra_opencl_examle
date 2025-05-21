/** @file cv_demo.cpp
 *
 * OpenCL example - convolution on the image
 *
 * @author Dominik Belter
 */

#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

extern "C" {
#include "NNets/nnets_conv_cl.h"
}

/// converts cv::Mat to vector
std::vector<unsigned char> toVector(cv::Mat image){
    std::vector<unsigned char> array((unsigned char*)image.data, (unsigned char*)image.data + image.rows * image.cols);
    return array;
}

/// converts vector to cv::Mat
void fromVector(std::vector<unsigned char> array, cv::Mat& image){
    std::vector<uchar> pv(array.size(),0);
    for(unsigned int i = 0; i < array.size(); i++) {
        pv[i] = (uchar) array.at(i);
    }
    if(array.size() == (size_t)image.rows*image.cols){
        memcpy(image.data, &pv.front(), array.size()*sizeof(uchar));
    }
}

int main()
{
    try {
        cv::Mat image;
        // load image
        image = cv::imread("../../resources/messor2.jpg", cv::IMREAD_COLOR);
        //        cv::GaussianBlur( image, image, cv::Size( 21, 21 ), 5, 5 );
        cv::Mat gray;
        // Convert the image to grayscale
        cv::cvtColor(image, gray, cv::COLOR_BGRA2GRAY );

        std::vector<unsigned char> input = toVector(gray);

        cv::namedWindow( "Gray", cv::WINDOW_AUTOSIZE );// Create a window for display.
        cv::imshow( "Gray", gray );

        // prepare output 1D vector (vectorized image)
        size_t vectorSize = input.size();
        std::vector<unsigned char> output(vectorSize,100);
        std::vector<char> mask = {-1,0,1,-1,0,1,-1,0,1};
        std::vector<char> mask_cl = {-1,0,1,-1,0,1,-1,0,1};
        // initialize opencl
        initializeOpenCLConv(image.rows, image.cols);

        //compute convolution using OpenCL
        std::chrono::steady_clock::time_point beginCL = std::chrono::steady_clock::now();
        computeoutputConv(&input.front(), &mask_cl.front(), &output.front(), gray.rows, gray.cols);
        std::chrono::steady_clock::time_point endCL = std::chrono::steady_clock::now();
        std::cout << "Time difference for OpenCL= " <<
                     std::chrono::duration_cast<std::chrono::microseconds>(endCL - beginCL).count() << "[µs]\n";
        fromVector(output, gray);
        cv::namedWindow( "Result CL", cv::WINDOW_AUTOSIZE );// Create a window for display.
        cv::imshow( "Result CL", gray );

        //compute convolution using CPU
        std::chrono::steady_clock::time_point beginCPU = std::chrono::steady_clock::now();
        for(size_t i = 0; i < vectorSize; i++)
        {
            conv_c(&input.front(), &mask.front(), &output.front(), gray.rows, gray.cols, i);
        }
        std::chrono::steady_clock::time_point endCPU = std::chrono::steady_clock::now();
        std::cout << "Time difference for CPU= " <<
                     std::chrono::duration_cast<std::chrono::microseconds>(endCPU - beginCPU).count() << "[µs]\n";
        fromVector(output, gray);
        cv::namedWindow( "Result CPU", cv::WINDOW_AUTOSIZE );// Create a window for display.
        cv::imshow( "Result CPU", gray );

        cv::waitKey(0);
    }
    catch (const std::exception& ex) {
        std::cerr << ex.what() << std::endl;
        return 1;
    }
    return 0;
}
