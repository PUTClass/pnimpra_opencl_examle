#define __CL_ENABLE_EXCEPTIONS
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/cl.hpp>
#else
#include <CL/opencl.hpp>
#endif
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <vector>

int main(void)
{
    try {
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);
        if (platforms.size() == 0) {
            std::cout << "No platforms\n";
            return -1;
        }

        // Print the number of platforms
        std::cout << "Detected platforms no: " << platforms.size() << std::endl;
        std::string platformVendor;
        for (unsigned int i = 0; i < platforms.size(); ++i) {
            platforms[i].getInfo((cl_platform_info)CL_PLATFORM_VENDOR, &platformVendor);
            std::cout << "Vendor: " << platformVendor << std::endl;
        }

        cl_context_properties properties[] =
        {
                CL_CONTEXT_PLATFORM,
                (cl_context_properties)(platforms[0])(),
                0
        };
        cl::Context context(CL_DEVICE_TYPE_ALL, properties);

        std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();

        std::cout << "Device no: " << devices.size() << std::endl;
        for (unsigned int i = 0; i < devices.size(); ++i) {
            std::cout << "Device #" << i << ": " << devices[i].getInfo<CL_DEVICE_NAME>() << std::endl;
        }
    }
    catch (cl::Error& err) {
        std::cerr << "ERROR: " << err.what() << "(" << err.err() << ")\n";
    }

    return EXIT_SUCCESS;

}
