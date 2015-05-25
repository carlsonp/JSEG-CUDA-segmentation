# Using Ryan's cmake modules to find OpenCV
# https://github.com/rpavlik/cmake-modules

cmake_minimum_required(VERSION 2.8.1)

# Set package properties
project(JSEG)

# Locally-developed modules dist'ed with this app - always have this first.
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
include(UseBackportedModules)

find_package(CUDA REQUIRED)
find_package(cutil REQUIRED)
include_directories(${CUTIL_INCLUDE_DIRS})

find_package(OpenCV REQUIRED)
include_directories(${OPENCV_INCLUDE_DIRS})

find_package(Boost REQUIRED system filesystem)
include_directories (${Boost_INCLUDE_DIRS})

set(SOURCES JSEG.cpp JSEG_kernel.cu JSEG_kernel.h deviceQuery.h JSEG_CPU.h)
cuda_add_executable(JSEG ${SOURCES})
target_link_libraries(JSEG cutil ${OPENCV_LIBRARIES} ${Boost_LIBRARIES})
