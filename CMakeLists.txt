cmake_minimum_required(VERSION 3.10)
project(HSS_HODLR_Benchmark)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# =========================================================
# macOS OpenMP Fix (针对 Apple Clang 的手动路径修复)
# =========================================================
if(APPLE)
    # 尝试寻找 Homebrew 安装的 libomp
    if(EXISTS "/opt/homebrew/opt/libomp") 
        set(OpenMP_ROOT "/opt/homebrew/opt/libomp")
    elseif(EXISTS "/usr/local/opt/libomp")
        set(OpenMP_ROOT "/usr/local/opt/libomp")
    endif()

    if(DEFINED OpenMP_ROOT)
        message(STATUS "Mac OpenMP Root found at: ${OpenMP_ROOT}")
        include_directories("${OpenMP_ROOT}/include")
        link_directories("${OpenMP_ROOT}/lib")
        
        # 强制添加编译和链接参数
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Xpreprocessor -fopenmp -I${OpenMP_ROOT}/include")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Xpreprocessor -fopenmp -I${OpenMP_ROOT}/include")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -L${OpenMP_ROOT}/lib -lomp")
    endif()
endif()

# =========================================================
# 查找依赖库
# =========================================================
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)
# 使用 QUIET 避免在 macOS 上因为 find_package 失败而报错（上面已手动处理）
find_package(OpenMP QUIET)

# 全局包含目录
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# =========================================================
# 目标 1: HODLR Benchmark (原来的实现)
# =========================================================
add_executable(hodlr_test src/main_hodlr.cpp)

# 链接 BLAS/LAPACK
if(BLAS_FOUND AND LAPACK_FOUND)
    target_link_libraries(hodlr_test PRIVATE ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})
endif()

# 链接 OpenMP
if(OpenMP_CXX_FOUND)
    target_link_libraries(hodlr_test PRIVATE OpenMP::OpenMP_CXX)
elseif(APPLE)
    # macOS 手动链接回退
    target_link_libraries(hodlr_test PRIVATE omp)
endif()

# =========================================================
# 目标 2: HSS Benchmark (新的 Nested Basis 实现)
# =========================================================
add_executable(hss_test src/main_hss.cpp)

# 链接 BLAS/LAPACK
if(BLAS_FOUND AND LAPACK_FOUND)
    target_link_libraries(hss_test PRIVATE ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})
endif()

# 链接 OpenMP
if(OpenMP_CXX_FOUND)
    target_link_libraries(hss_test PRIVATE OpenMP::OpenMP_CXX)
elseif(APPLE)
    # macOS 手动链接回退
    target_link_libraries(hss_test PRIVATE omp)
endif()