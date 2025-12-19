#include <iostream>
#include <chrono>
#include <omp.h>
#include <iomanip>

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

double calc_err(const Matrix& ref, const Matrix& res) {
    double n=0, d=0;
    for(int i=0; i<ref.rows(); ++i) {
        double diff = ref.data()[i] - res.data()[i];
        n += diff*diff; d += ref.data()[i]*ref.data()[i];
    }
    return sqrt(n/d);
}

int main() {
    int N = 2048*2;
    int leaf_size = 256*2;
    double tol = 1e-6; // Strict tolerance for direct solver check

    cout << "=== HSS Benchmark (Direct Solver) ===" << endl;
    cout << "N = " << N << endl;

    // Use standard Cauchy matrix (ill-conditioned)
    Matrix A = Kernels::generate_matrix(Kernels::CAUCHY, N);
    
    Matrix x = Matrix::Random(N, 1);
    Matrix b = A * x;

    // --- 1. Serial HSS ---
    cout << "\n[1] Serial HSS" << endl;
    HSSMatrix hss(tol);
    Timer t;
    hss.build_from_dense(A, leaf_size);
    double t_ser_build = t.elapsed();
    cout << "Build: " << t_ser_build << "s" << endl;

    t.reset();
    Matrix y_ser = hss.multiply(x);
    cout << "MV Time: " << t.elapsed() << "s | Err: " << calc_err(b, y_ser) << endl;

    t.reset();
    Matrix x_ser = hss.solve(b);
    cout << "Solve Time: " << t.elapsed() << "s | Err: " << calc_err(x, x_ser) << endl;

    // --- 2. Parallel HSS ---
    cout << "\n[2] Parallel HSS (OpenMP)" << endl;
    HSSMatrixOMP hss_omp(tol);
    t.reset();
    hss_omp.build_from_dense(A, leaf_size);
    double t_omp_build = t.elapsed();
    cout << "Build: " << t_omp_build << "s [Speedup: " << t_ser_build/t_omp_build << "x]" << endl;

    t.reset();
    Matrix y_omp = hss_omp.multiply(x);
    cout << "MV Time: " << t.elapsed() << "s | Err: " << calc_err(b, y_omp) << endl;

    t.reset();
    Matrix x_omp = hss_omp.solve(b);
    cout << "Solve Time: " << t.elapsed() << "s | Err: " << calc_err(x, x_omp) << endl;

    return 0;
}