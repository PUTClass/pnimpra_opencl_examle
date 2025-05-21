# cmake requirements
cmake_minimum_required(VERSION 2.6)

# Build options have to be before PROJECT(...)
SET(CMAKE_CONFIGURATION_TYPES "Release" CACHE PATH "Configuration types")
SET(CMAKE_BUILD_TYPE "Release" CACHE PATH "Current build configuration")

# Example Project configuration
PROJECT(OPENCL_EXAMPLE)

# Compiler flags
if (CMAKE_COMPILER_IS_GNUCC)
    set (CMAKE_CXX_FLAGS "-D_REETRANT -Wall -Wextra -pedantic -Wno-long-long")
    if (CMAKE_BUILD_TYPE STREQUAL "Debug")
        set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ggdb -O0")
    elseif( CMAKE_BUILD_TYPE STREQUAL "Release" )
        set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DNDEBUG -O3 -fno-strict-aliasing")
    endif ()
endif (CMAKE_COMPILER_IS_GNUCC)

# Folders
SET_PROPERTY(GLOBAL PROPERTY USE_FOLDERS ON)

# Settings
INCLUDE_DIRECTORIES("${CMAKE_SOURCE_DIR}")
INCLUDE_DIRECTORIES("${CMAKE_SOURCE_DIR}/include")

find_package(OpenCL REQUIRED)
include_directories(${OPENCL_INCLUDE_DIR})

FIND_PACKAGE(OpenCV REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS} )

# Executable output directory
SET(RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/build/bin${OUTPUT_DIRECTORY_POSTFIX} CACHE PATH "Executable output directory")
mark_as_advanced(RUNTIME_OUTPUT_DIRECTORY)

# Dynamic library output directory
SET(LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/build/bin${OUTPUT_DIRECTORY_POSTFIX} CACHE PATH "Dynamic library output directory")
mark_as_advanced(LIBRARY_OUTPUT_DIRECTORY)

# Static library output directory
SET(ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/build/lib${OUTPUT_DIRECTORY_POSTFIX} CACHE PATH "Static library output directory")
mark_as_advanced(ARCHIVE_OUTPUT_DIRECTORY)

SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${RUNTIME_OUTPUT_DIRECTORY})
SET(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${LIBRARY_OUTPUT_DIRECTORY})
SET(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${ARCHIVE_OUTPUT_DIRECTORY})
foreach(CONFIGURATION_TYPE ${CMAKE_CONFIGURATION_TYPES})
    string(TOUPPER ${CONFIGURATION_TYPE} CONFIGURATION_TYPE)
    SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${CONFIGURATION_TYPE} ${RUNTIME_OUTPUT_DIRECTORY})
    SET(CMAKE_LIBRARY_OUTPUT_DIRECTORY_${CONFIGURATION_TYPE} ${LIBRARY_OUTPUT_DIRECTORY})
    SET(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_${CONFIGURATION_TYPE} ${ARCHIVE_OUTPUT_DIRECTORY})
endforeach(CONFIGURATION_TYPE CMAKE_CONFIGURATION_TYPES)

if(MSVC)
    SET (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc")
endif()

# Offer the user the choice of overriding the installation directories
set(INSTALL_LIB_DIR lib CACHE PATH "Installation directory for libraries")
set(INSTALL_BIN_DIR bin CACHE PATH "Installation directory for executables")
set(INSTALL_INCLUDE_DIR include CACHE PATH
    "Installation directory for header files")
if(WIN32 AND NOT CYGWIN)
    set(DEF_INSTALL_CMAKE_DIR CMake)
else()
    set(DEF_INSTALL_CMAKE_DIR CMake)
endif()
set(INSTALL_CMAKE_DIR ${DEF_INSTALL_CMAKE_DIR} CACHE PATH "Installation directory for CMake files")


###############################################################################
#
# Opencl demos
#
###############################################################################

message(STATUS ${OPENCL_LIBRARIES})
message(STATUS ${OPENCL_INCLUDE_DIR})

# Compiler flags
if (CMAKE_COMPILER_IS_GNUCC)
    set (CMAKE_CXX_FLAGS "-D_REETRANT -Wall -Wextra -pedantic -Wno-long-long")
    if (CMAKE_BUILD_TYPE STREQUAL "Debug")
        set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ggdb -O0")
    elseif( CMAKE_BUILD_TYPE STREQUAL "Release" )
        set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DNDEBUG -O3 -fno-strict-aliasing")
    endif ()
endif (CMAKE_COMPILER_IS_GNUCC)

#NNets library
file(GLOB NNETS_SOURCES
    "${CMAKE_SOURCE_DIR}/src/NNets/*.cc"
    )
file(GLOB NNETS_HEADERS
    "${CMAKE_SOURCE_DIR}/include/NNets/*.h"
    )
ADD_LIBRARY(nnets STATIC ${NNETS_SOURCES} ${NNETS_HEADERS})
TARGET_LINK_LIBRARIES(nnets OpenCL)

SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-rpath ./")
SET(DEMO_SOURCES ./main.cpp)
ADD_EXECUTABLE(openclc ${DEMO_SOURCES})
TARGET_LINK_LIBRARIES(openclc)

SET(DEMO_SOURCES ./demos/opencl_demo.cc)
ADD_EXECUTABLE(opencl_demo ${DEMO_SOURCES})
TARGET_LINK_LIBRARIES(opencl_demo OpenCL)

SET(DEMO_SOURCES ./demos/opencl_hello.cc)
ADD_EXECUTABLE(opencl_hello ${DEMO_SOURCES})
TARGET_LINK_LIBRARIES(opencl_hello OpenCL)

SET(DEMO_SOURCES ./demos/cpp_demo.cpp)
ADD_EXECUTABLE(cpp_demo ${DEMO_SOURCES})
TARGET_LINK_LIBRARIES(cpp_demo nnets)

SET(DEMO_SOURCES ./demos/cv_demo.cpp)
ADD_EXECUTABLE(cv_demo ${DEMO_SOURCES})
TARGET_LINK_LIBRARIES(cv_demo nnets ${OpenCV_LIBS})
