cmake_minimum_required(VERSION 3.10)
project(HSS_Benchmark)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(APPLE)
    set(OpenMP_ROOT "/opt/homebrew/opt/libomp")
endif()
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)

find_package(OpenMP REQUIRED)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

set(SOURCES src/main.cpp)

add_executable(hss_test ${SOURCES})

if(BLAS_FOUND AND LAPACK_FOUND)
    target_link_libraries(hss_test PRIVATE 
        ${BLAS_LIBRARIES} 
        ${LAPACK_LIBRARIES}
    )
endif()

if(OpenMP_CXX_FOUND)
    target_link_libraries(hss_test PRIVATE OpenMP::OpenMP_CXX)
endif()