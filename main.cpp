#include <iostream>

int main()
{
    try {
        std::cout << "OpenCL examples. Run demo programs:           \n";
        std::cout << "opencl_hello - read devices info              \n";
        std::cout << "opencl_demo - leaky ReLU CPU vs GPU           \n";
        std::cout << "cpp_demo - leaky ReLU CPU vs GPU using C++    \n";
        std::cout << "cv_demo - convolution CPU vs GPU using C++    \n";
    }
    catch (const std::exception& ex) {
        std::cerr << ex.what() << std::endl;
        return 1;
    }
    return 0;
}
