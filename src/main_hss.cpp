#include <iostream>
#include <chrono>
#include <omp.h>
#include <iomanip>
#include <cmath>

#include "../include/hss_routines.h"      
#include "../include/hss_routines_omp.h"  
#include "../include/kernel.h"

using namespace std;

class Timer {
    using clock_t = std::chrono::high_resolution_clock;
    std::chrono::time_point<clock_t> start_time;
public:
    Timer() : start_time(clock_t::now()) {}
    double elapsed() {
        auto end = clock_t::now();
        return std::chrono::duration<double>(end - start_time).count();
    }
    void reset() { start_time = clock_t::now(); }
};

// 使用相对误差 (Relative Error) 以便与 HODLR benchmark 统一
double calc_rel_error(const Matrix& ref, const Matrix& res) {
    double num = 0.0, den = 0.0;
    for(int i=0; i<ref.rows() * ref.cols(); ++i) {
        double d = ref.data()[i] - res.data()[i];
        num += d*d;
        den += ref.data()[i] * ref.data()[i];
    }
    if (den == 0) return 0.0;
    return std::sqrt(num) / std::sqrt(den);
}

int main() {
    int N = 2048;          // 保持你之前的设置
    int leaf_size = 256;
    double tol = 1e-9; 

    // 设置嵌套并行最大深度，防止任务过多 (针对 OpenMP Tasking)
    omp_set_max_active_levels(4);

    std::cout << "==========================================================" << std::endl;
    std::cout << " HSS Final Benchmark: Serial vs Parallel" << std::endl;
    std::cout << " N = " << N << " | Threads: " << omp_get_max_threads() << std::endl;
    std::cout << "==========================================================" << std::endl;

    // 为了数值稳定性，HSS 建议使用 Gaussian 核或者对角占优的矩阵
    std::cout << "\n>>> Generating Matrix (Gaussian)..." << std::endl;
    Matrix A = Kernels::generate_gaussian(N, 0.5, 1e-6); 
    
    Matrix x_true = Matrix::Random(N, 1);
    Matrix b = A * x_true;

    // --- 1. BLAS Baseline ---
    std::cout << "\n--- [1] Baseline: BLAS Dense ---" << std::endl;
    Timer t;
    
    // GEMV Baseline
    Matrix b_blas = A * x_true; 
    double t_blas_mv = t.elapsed();

    // Direct Solve Baseline (LU)
    // 注意：solve_lu 会修改原矩阵，所以需要深拷贝
    Matrix A_dense = A; 
    Matrix x_blas = b; 
    t.reset();
    A_dense.solve_lu(x_blas);
    double t_blas_solve = t.elapsed();

    std::cout << " GEMV: " << t_blas_mv << "s | Solve: " << t_blas_solve << "s" << std::endl;

    // --- 2. Serial HSS ---
    std::cout << "\n--- [2] Serial HSS ---" << std::endl;
    HSSMatrix hss(tol);
    
    t.reset();
    hss.build_from_dense(A, leaf_size);
    double t_ser_build = t.elapsed();
    std::cout << " Build: " << t_ser_build << "s" << std::endl;

    t.reset();
    Matrix y_ser = hss.multiply(x_true);
    double t_ser_mv = t.elapsed();
    std::cout << " GEMV:  " << t_ser_mv << "s | Err: " << calc_rel_error(b, y_ser) << std::endl;

    t.reset();
    Matrix x_ser = hss.solve(b);
    double t_ser_solve = t.elapsed();
    std::cout << " Solve: " << t_ser_solve << "s | Err: " << calc_rel_error(x_true, x_ser) << std::endl;

    // --- 3. Parallel HSS ---
    std::cout << "\n--- [3] Parallel HSS (OpenMP Tasking) ---" << std::endl;
    HSSMatrixOMP hss_omp(tol);
    
    t.reset();
    hss_omp.build_from_dense(A, leaf_size);
    double t_omp_build = t.elapsed();
    std::cout << " Build: " << t_omp_build << "s" << std::endl;

    t.reset();
    Matrix y_omp = hss_omp.multiply(x_true);
    double t_omp_mv = t.elapsed();
    std::cout << " GEMV:  " << t_omp_mv << "s | Err: " << calc_rel_error(b, y_omp) << std::endl;

    t.reset();
    Matrix x_omp = hss_omp.solve(b);
    double t_omp_solve = t.elapsed();
    std::cout << " Solve: " << t_omp_solve << "s | Err: " << calc_rel_error(x_true, x_omp) << std::endl;

    // --- Summary ---
    std::cout << "\n==========================================================" << std::endl;
    std::cout << ">>> Speedup Analysis <<<" << std::endl;
    std::cout << "Build Parallel Speedup: " << std::fixed << std::setprecision(2) << t_ser_build / t_omp_build << "x" << std::endl;
    std::cout << "Solve Parallel Speedup: " << t_ser_solve / t_omp_solve << "x" << std::endl;
    std::cout << "HSS Solve vs BLAS (Speedup): " << t_blas_solve / t_omp_solve << "x" << std::endl;
    std::cout << "==========================================================" << std::endl;

    return 0;
}