#include <iostream>
#include <chrono>
#include <iomanip>
#include <vector>
#include <cmath>
#include <omp.h>

#include "../include/hodlr_routines.h"      // Serial (Optimized RSVD + Woodbury)
#include "../include/hodlr_routines_omp.h"  // Parallel (Optimized RSVD + Woodbury)
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
    int N = 4096;          
    int leaf_size = 256 * 2;   
    double tol = 1e-8;     
    
    std::cout << "==========================================================" << std::endl;
    std::cout << " HODLR Final Benchmark: Serial vs Parallel" << std::endl;
    std::cout << " Matrix: " << N << "x" << N << " (Cauchy)" << std::endl;
    std::cout << " Max Threads: " << omp_get_max_threads() << std::endl;
    std::cout << "==========================================================" << std::endl;

    std::cout << "\n>>> Generating Matrix..." << std::endl;
    Matrix A = Kernels::generate_matrix(Kernels::CAUCHY, N);
    Matrix x_true = Matrix::Random(N, 1);
    Matrix b = A * x_true;

    // --- 1. BLAS Baseline ---
    std::cout << "\n--- [1] Baseline: BLAS Dense ---" << std::endl;
    Timer t;
    Matrix b_blas = A * x_true; 
    double t_blas_mv = t.elapsed();
    
    Matrix A_dense = A; 
    Matrix x_blas = b; 
    t.reset();
    A_dense.solve_lu(x_blas);
    double t_blas_solve = t.elapsed();
    
    std::cout << " GEMV: " << t_blas_mv << "s | Solve: " << t_blas_solve << "s" << std::endl;

    // --- 2. Serial HODLR ---
    std::cout << "\n--- [2] Serial HODLR (RSVD + Woodbury) ---" << std::endl;
    HODLRMatrix hodlr_serial(tol);

    t.reset();
    hodlr_serial.build_from_dense(A, leaf_size);
    double t_serial_build = t.elapsed();
    std::cout << " Build: " << t_serial_build << "s" << std::endl;

    t.reset();
    Matrix y_serial = hodlr_serial.multiply(x_true);
    double t_serial_mv = t.elapsed();
    double err_serial_mv = calc_rel_error(b, y_serial);
    std::cout << " GEMV:  " << t_serial_mv << "s | Err: " << err_serial_mv << std::endl;

    t.reset();
    Matrix x_serial = hodlr_serial.solve(b);
    double t_serial_solve = t.elapsed();
    double err_serial_solve = calc_rel_error(x_true, x_serial);
    std::cout << " Solve: " << t_serial_solve << "s | Err: " << err_serial_solve << std::endl;

    // --- 3. Parallel HODLR ---
    std::cout << "\n--- [3] Parallel HODLR (RSVD + Woodbury + OpenMP) ---" << std::endl;
    HODLRMatrixOMP hodlr_omp(tol);

    t.reset();
    hodlr_omp.build_from_dense(A, leaf_size);
    double t_omp_build = t.elapsed();
    std::cout << " Build: " << t_omp_build << "s" << std::endl;

    t.reset();
    Matrix y_omp = hodlr_omp.multiply(x_true);
    double t_omp_mv = t.elapsed();
    double err_omp_mv = calc_rel_error(b, y_omp);
    std::cout << " GEMV:  " << t_omp_mv << "s | Err: " << err_omp_mv << std::endl;

    t.reset();
    Matrix x_omp = hodlr_omp.solve(b);
    double t_omp_solve = t.elapsed();
    double err_omp_solve = calc_rel_error(x_true, x_omp);
    std::cout << " Solve: " << t_omp_solve << "s | Err: " << err_omp_solve << std::endl;
    
    // --- Summary ---
    std::cout << "\n==========================================================" << std::endl;
    std::cout << ">>> Speedup Analysis <<<" << std::endl;
    std::cout << "Build Parallel Speedup: " << std::fixed << std::setprecision(2) << t_serial_build / t_omp_build << "x" << std::endl;
    std::cout << "Solve Parallel Speedup: " << t_serial_solve / t_omp_solve << "x" << std::endl;
    std::cout << "HODLR Solve vs BLAS (Speedup): " << t_blas_solve / t_omp_solve << "x" << std::endl;
    std::cout << "==========================================================" << std::endl;

    return 0;
}