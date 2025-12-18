#include <iostream>
#include <chrono>
#include <iomanip>
#include <vector>
#include <cmath>

#include "../include/hss_routines.h"      // Serial
#include "../include/hss_routines_omp.h"  // Parallel (OpenMP)
#include "../include/kernel.h"            // Kernel Matrices

using namespace std;

// --- Timer Utility ---
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

// --- Helper for Relative Error ---
double calc_rel_error(const Matrix& ref, const Matrix& res) {
    double num = 0.0;
    double den = 0.0;
    for(int i=0; i<ref.rows() * ref.cols(); ++i) {
        double d = ref.data()[i] - res.data()[i];
        num += d*d;
        den += ref.data()[i] * ref.data()[i];
    }
    if (den == 0) return 0.0;
    return std::sqrt(num) / std::sqrt(den);
}

int main() {
    // ==========================================
    // CONFIGURATION
    // ==========================================
    int N = 4096;           // Matrix Dimension (Needs to be large for HSS to win)
    int leaf_size = 256;    // Block size
    double tol = 1e-12;      // Compression tolerance
    
    // Choose Kernel: CAUCHY, GAUSSIAN, COULOMB
    Kernels::Type kernel_type = Kernels::CAUCHY; 
    string kernel_name = "Cauchy";

    // ==========================================
    
    std::cout << "==========================================================" << std::endl;
    std::cout << " HSS Benchmark: BLAS vs Serial vs OpenMP" << std::endl;
    std::cout << " Matrix Type: " << kernel_name << " (Low-rank off-diagonals)" << std::endl;
    std::cout << " N = " << N << ", Leaf Size = " << leaf_size << ", Tol = " << tol << std::endl;
    #ifdef _OPENMP
    std::cout << " OpenMP Enabled. Max Threads: " << omp_get_max_threads() << std::endl;
    #else
    std::cout << " OpenMP DISABLED." << std::endl;
    #endif
    std::cout << "==========================================================\n" << std::endl;

    // 1. Generate Data-Sparse Matrix
    std::cout << "[Init] Generating Kernel Matrix..." << std::endl;
    Matrix A = Kernels::generate_matrix(kernel_type, N);
    
    // RHS Vector
    Matrix x_true = Matrix::Random(N, 1);
    Matrix b = A * x_true; // b = A * x_true

    // ---------------------------------------------------------
    // BASELINE: BLAS / LAPACK
    // ---------------------------------------------------------
    std::cout << "--- [1] Baseline: BLAS / LAPACK Dense ---" << std::endl;
    Timer t;
    
    // GEMV Baseline
    Matrix b_blas = A * x_true; 
    double time_blas_gemv = t.elapsed();
    std::cout << "  GEMV Time:   " << std::fixed << std::setprecision(6) << time_blas_gemv << " s" << std::endl;

    // Solver Baseline (LU)
    // We work on a copy because solve_lu is in-place
    Matrix A_dense = A; 
    Matrix x_blas = b; 
    
    t.reset();
    A_dense.solve_lu(x_blas); 
    double time_blas_solve = t.elapsed();
    
    std::cout << "  Solve Time:  " << std::setw(10) << time_blas_solve << " s" << std::endl;
    std::cout << "  Error (L2):  " << std::scientific << calc_rel_error(x_true, x_blas) << std::endl;
    std::cout << std::endl;

    // ---------------------------------------------------------
    // SERIAL HSS
    // ---------------------------------------------------------
    std::cout << "--- [2] Serial HSS Implementation ---" << std::endl;
    HSSMatrix hss_serial(tol);
    
    // Build
    t.reset();
    hss_serial.build_from_dense(A, leaf_size);
    double time_serial_build = t.elapsed();
    std::cout << "  Build Time:  " << std::fixed << std::setprecision(6) << time_serial_build << " s" << std::endl;

    // GEMV
    t.reset();
    Matrix y_serial = hss_serial.multiply(x_true);
    double time_serial_gemv = t.elapsed();
    double err_serial_gemv = calc_rel_error(b, y_serial);
    std::cout << "  GEMV Time:   " << std::setw(10) << time_serial_gemv << " s | Err: " << err_serial_gemv << std::endl;

    // Solve (Preconditioner/Approx)
    t.reset();
    Matrix x_serial_sol = hss_serial.solve(b);
    double time_serial_solve = t.elapsed();
    double err_serial_solve = calc_rel_error(x_true, x_serial_sol); 
    std::cout << "  Solve Time:  " << std::setw(10) << time_serial_solve << " s | Err: " << err_serial_solve << std::endl;
    std::cout << std::endl;

    // ---------------------------------------------------------
    // PARALLEL HSS (OpenMP)
    // ---------------------------------------------------------
    std::cout << "--- [3] Parallel HSS (Level-by-Level OpenMP) ---" << std::endl;
    HSSMatrixOMP hss_omp(tol);

    // Build
    t.reset();
    hss_omp.build_from_dense(A, leaf_size);
    double time_omp_build = t.elapsed();
    
    std::cout << "  Build Time:  " << std::fixed << std::setprecision(6) << time_omp_build << " s";
    std::cout << " [Speedup vs Serial: " << std::fixed << std::setprecision(2) << time_serial_build / time_omp_build << "x]" << std::endl;

    // GEMV
    t.reset();
    Matrix y_omp = hss_omp.multiply(x_true);
    double time_omp_gemv = t.elapsed();
    double err_omp_gemv = calc_rel_error(b, y_omp);
    std::cout << "  GEMV Time:   " << std::setw(10) << time_omp_gemv << " s";
    std::cout << " [Speedup vs Serial: " << std::fixed << std::setprecision(2) << time_serial_gemv / time_omp_gemv << "x]";
    std::cout << " | Err: " << std::scientific << err_omp_gemv << std::endl;

    // Solve
    t.reset();
    Matrix x_omp_sol = hss_omp.solve(b);
    double time_omp_solve = t.elapsed();
    double err_omp_solve = calc_rel_error(x_true, x_omp_sol);
    std::cout << "  Solve Time:  " << std::setw(10) << time_omp_solve << " s";
    std::cout << " [Speedup vs Serial: " << std::fixed << std::setprecision(2) << time_serial_solve / time_omp_solve << "x]";
    std::cout << " | Err: " << std::scientific << err_omp_solve << std::endl;

    std::cout << "\n==========================================================" << std::endl;
    
    return 0;
}