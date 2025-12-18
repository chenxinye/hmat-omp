#include <iostream>
#include <chrono>
#include "../include/hss_routines.h" 

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

int main() {
    srand(42);
    int N = 512;
    int leaf_size = 64;
    
    std::cout << "=== HSS C++ Implementation (utils/hss_routines) ===" << std::endl;

    // 1. Create Data
    // Diagonally dominant for stable LU
    Matrix A = Matrix::Random(N, N);
    for(int i=0; i<N; ++i) A(i,i) += 10.0; 
    
    Matrix x = Matrix::Random(N, 1);

    // 2. Build HSS
    HSSMatrix hss(1e-6);
    Timer t;
    hss.build_from_dense(A, leaf_size);
    std::cout << "Build Time: " << t.elapsed() << "s" << std::endl;

    // 3. Test GEMV
    t.reset();
    Matrix y_hss = hss.multiply(x);
    std::cout << "HSS GEMV Time: " << t.elapsed() << "s" << std::endl;

    // 4. Test Dense Solver (Internal wrapper check)
    std::cout << "\n[Testing Internal LU Wrapper]..." << std::endl;
    Matrix b = A * x;
    Matrix x_solved = b;
    
    t.reset();
    A.solve_lu(x_solved); 
    std::cout << "Direct Dense LU Solve Time: " << t.elapsed() << "s" << std::endl;
    
    double err = 0;
    for(int i=0; i<N; ++i) {
        double d = x_solved(i,0) - x(i,0);
        err += d*d;
    }
    std::cout << "Dense Solve Error (L2): " << std::sqrt(err) << std::endl;

    return 0;
}