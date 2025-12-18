#include <iostream>
#include <chrono>
#include <iomanip>
#include <vector>
#include <cmath>

#include "../include/hss_routines.h"      // Serial
#include "../include/hss_routines_omp.h"  // Parallel (OpenMP)

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

// --- Helper for Diff ---
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
    // Configuration
    int N = 2048 * 2;          // Matrix Dimension (Increase to see OMP benefits)
    int leaf_size = 256 * 2;   // Block size
    double tol = 1e-8;     // Compression tolerance

    std::cout << "==========================================================" << std::endl;
    std::cout << " HSS Benchmark: BLAS vs Serial vs OpenMP" << std::endl;
    std::cout << " N = " << N << ", Leaf Size = " << leaf_size << std::endl;
    std::cout << " Max Threads: " << omp_get_max_threads() << std::endl;
    std::cout << "==========================================================\n" << std::endl;

    // 1. Prepare Data
    std::cout << "[Init] Generating Data..." << std::endl;
    // Diagonally dominant for stable LU solve testing
    Matrix A = Matrix::Random(N, N);
    for(int i=0; i<N; ++i) A(i,i) += (double)N; 
    
    Matrix x = Matrix::Random(N, 1);
    Matrix b_ref; // To be computed

    // ---------------------------------------------------------
    // BASELINE: BLAS / LAPACK (Trivial)
    // ---------------------------------------------------------
    std::cout << "--- [1] Baseline: BLAS / LAPACK Dense ---" << std::endl;
    Timer t;
    
    // GEMV Baseline
    b_ref = A * x; 
    double time_blas_gemv = t.elapsed();
    std::cout << "  GEMV Time:   " << std::setw(10) << time_blas_gemv << " s" << std::endl;

    // Solver Baseline (LU)
    t.reset();
    Matrix x_blas_sol = b_ref;
    A.solve_lu(x_blas_sol); // A is modified? No, solve_lu modifies 'this' data? 
    // Wait, matrix.h solve_lu performs LU in-place on 'data_'. 
    // We need a copy of A for dense solve if we want to keep A for HSS build!
    // Let's reload A or use a copy.
    Matrix A_dense = A; // Copy
    Timer t_sol;
    A_dense.solve_lu(x_blas_sol); 
    double time_blas_solve = t_sol.elapsed();
    
    std::cout << "  Solve Time:  " << std::setw(10) << time_blas_solve << " s" << std::endl;
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
    std::cout << "  Build Time:  " << std::setw(10) << time_serial_build << " s" << std::endl;

    // GEMV
    t.reset();
    Matrix y_serial = hss_serial.multiply(x);
    double time_serial_gemv = t.elapsed();
    double err_serial_gemv = calc_rel_error(b_ref, y_serial);
    std::cout << "  GEMV Time:   " << std::setw(10) << time_serial_gemv << " s | Err: " << err_serial_gemv << std::endl;

    // Solve (Preconditioner Style)
    t.reset();
    Matrix x_serial_sol = hss_serial.solve(b_ref);
    double time_serial_solve = t.elapsed();
    // Compare HSS solution with Dense solution (or true x)
    double err_serial_solve = calc_rel_error(x, x_serial_sol); 
    std::cout << "  Solve Time:  " << std::setw(10) << time_serial_solve << " s | Err: " << err_serial_solve << " (Approx)" << std::endl;
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
    std::cout << "  Build Time:  " << std::setw(10) << time_omp_build << " s";
    std::cout << " [Speedup: " << std::fixed << std::setprecision(2) << time_serial_build / time_omp_build << "x]" << std::endl;

    // GEMV
    t.reset();
    Matrix y_omp = hss_omp.multiply(x);
    double time_omp_gemv = t.elapsed();
    double err_omp_gemv = calc_rel_error(b_ref, y_omp);
    std::cout << "  GEMV Time:   " << std::setw(10) << time_omp_gemv << " s";
    std::cout << " [Speedup: " << std::fixed << std::setprecision(2) << time_serial_gemv / time_omp_gemv << "x]";
    std::cout << " | Err: " << std::scientific << err_omp_gemv << std::endl;

    // Solve
    t.reset();
    Matrix x_omp_sol = hss_omp.solve(b_ref);
    double time_omp_solve = t.elapsed();
    double err_omp_solve = calc_rel_error(x, x_omp_sol);
    std::cout << "  Solve Time:  " << std::setw(10) << time_omp_solve << " s";
    std::cout << " [Speedup: " << std::fixed << std::setprecision(2) << time_serial_solve / time_omp_solve << "x]";
    std::cout << " | Err: " << std::scientific << err_omp_solve << std::endl;

    std::cout << "\n==========================================================" << std::endl;
    
    return 0;
}