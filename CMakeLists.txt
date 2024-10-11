cmake_minimum_required(VERSION 3.4)
set(PROJECT_NAME "REVO")
project(${PROJECT_NAME})

SET(CMAKE_BUILD_TYPE RELEASE)

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall -O1 -msse3 -msse4.1 -mavx2 -fopenmp")
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -O1 -msse3 -msse4.1 -mavx2 -fopenmp")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O3 -msse3 -msse4.1 -mavx2 -fopenmp")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -msse3 -msse4.1 -mavx2 -fopenmp")

include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++17" COMPILER_SUPPORTS_CXX17)
if(COMPILER_SUPPORTS_CXX17)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
    message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has c++17 support.")
endif()

# Initialize variables to hold all source and header files
set(FILE_DIRS system utils datastructures io)
set(ALL_SOURCES "main.cpp")
set(ALL_HEADERS "")

# OpenCV
find_package(OpenCV 3.4.16 REQUIRED PATHS /opt/opencv/34)
if(NOT OpenCV_FOUND)
    message(FATAL_ERROR "OpenCV > 3.4.16 not found.")
endif()

# Boost
find_package(Boost COMPONENTS system filesystem REQUIRED)

# CUDA
find_package(CUDA REQUIRED)
message("CUDA_INCLUDE_DIRS: ${CUDA_INCLUDE_DIRS}")
include_directories(${CUDA_INCLUDE_DIRS})

# cuDNN
set(CUDNN_DIR /opt/cudnn/cudnn-linux-x86_64-9.4.0.58_cuda12-archive)
include_directories(${CUDNN_DIR}/include)
file(GLOB CUDNN_LIBRARIES "${TENSORRT_LIBRARY_DIR}/*.so")

# TensorRT
set(TENSORRT_DIR /opt/tensorrt/CUDA12/TensorRT-10.4.0.26)
include_directories(${TENSORRT_DIR}/include)
file(GLOB TENSORRT_LIBRARIES "${TENSORRT_DIR}/lib/*.so")

# Eigen
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIRS})

# Sophus
include_directories("thirdparty/Sophus")

# Pangolin
OPTION(WITH_PANGOLIN_VIEWER "Build with Pangolin viewer?" ON)
if (WITH_PANGOLIN_VIEWER)
    set(Pangolin_USER_DIR /opt/pangolin/060)
    find_package(Pangolin REQUIRED PATHS ${Pangolin_USER_DIR})
    include_directories(${Pangolin_USER_DIR}/include)
    add_definitions(-DWITH_PANGOLIN_VIEWER)
    list(APPEND LIBS ${Pangolin_LIBRARIES})
    list(APPEND FILE_DIRS "gui")
endif()

# LibRealSense
OPTION(WITH_REALSENSE "Build with RealSense Sensor?" OFF)

# Iterate over each directory in FILE_DIRS
message("FILE_DIRS: ${FILE_DIRS}")
foreach(DIR ${FILE_DIRS})
    # Collect all source files in the src directory
    file(GLOB_RECURSE SOURCES "${DIR}/*.cpp" "${DIR}/*.cu" "${DIR}/*.cc")
    list(APPEND ALL_SOURCES ${SOURCES})

    # Collect all header files in the include directory
    file(GLOB_RECURSE HEADERS "${DIR}/*.h" "${DIR}/*.hpp")
    list(APPEND ALL_HEADERS ${HEADERS})

    # Add include directories
    include_directories(${DIR})
endforeach()

add_executable(${PROJECT_NAME} ${ALL_SOURCES} ${ALL_HEADERS})

target_link_libraries(${PROJECT_NAME}
    ${LIBS}
    ${OpenCV_LIBS} 
    ${Boost_FILESYSTEM_LIBRARY} ${Boost_SYSTEM_LIBRARY}
    ${CUDA_LIBRARIES} ${CUDNN_LIBRARIES} ${TENSORRT_LIBRARIES}
)
