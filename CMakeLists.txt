cmake_minimum_required(VERSION 3.10)
project(HSS_Solver_CPP)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find BLAS and LAPACK
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Source files
set(SOURCES src/main.cpp)

# Executable
add_executable(hss_test ${SOURCES})

# Link Libraries
if(BLAS_FOUND AND LAPACK_FOUND)
    target_link_libraries(hss_test PRIVATE ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})
else()
    message(FATAL_ERROR "BLAS or LAPACK not found!")
endif()