cmake_minimum_required(VERSION 3.10)
project(HSS_Benchmark)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(APPLE)
    set(OpenMP_ROOT "/opt/homebrew/opt/libomp")
    include_directories("/opt/homebrew/opt/libomp/include")
    link_directories("/opt/homebrew/opt/libomp/lib")

    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include")
    
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -L/opt/homebrew/opt/libomp/lib -lomp")
endif()

find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)

find_package(OpenMP QUIET)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

set(SOURCES src/main.cpp)

add_executable(hss_test ${SOURCES})

if(BLAS_FOUND AND LAPACK_FOUND)
    target_link_libraries(hss_test PRIVATE ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})
endif()

if(OpenMP_CXX_FOUND)
    target_link_libraries(hss_test PRIVATE OpenMP::OpenMP_CXX)
elseif(APPLE)
    target_link_libraries(hss_test PRIVATE omp)
endif()