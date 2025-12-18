#ifndef KERNEL_H
#define KERNEL_H

#include "matrix.h"
#include <vector>
#include <cmath>
#include <algorithm>

namespace Kernels {

    enum Type {
        CAUCHY,
        GAUSSIAN,
        COULOMB
    };

    // Helper: Generate 1D grid points in [0, 1]
    inline std::vector<double> generate_grid(int N) {
        std::vector<double> points(N);
        for(int i = 0; i < N; ++i) {
            // Chebyshev nodes or uniform grid usually work well
            // Using uniform grid for simplicity
            points[i] = (double)i / (double)(N - 1);
        }
        return points;
    }

    // 1. Cauchy Matrix: A_ij = 1 / (x_i + y_j)
    // Highly compressible off-diagonals.
    inline Matrix generate_cauchy(int N, double diagonal_shift = 0.0) {
        Matrix A(N, N);
        std::vector<double> x = generate_grid(N);
        
        // Use a slightly shifted set for y to avoid singularity if diagonal_shift is 0
        // Or simply x_i + x_j + constant
        for(int j = 0; j < N; ++j) {
            for(int i = 0; i < N; ++i) {
                // Determine denominator
                double denom = x[i] + x[j] + 1.0; // +1 to keep it bounded
                A(i, j) = 1.0 / denom;
            }
            // Add diagonal shift to ensure positive definiteness (stability for LU/Cholesky)
            if (diagonal_shift != 0.0) {
                A(j, j) += diagonal_shift;
            }
        }
        return A;
    }

    // 2. Gaussian Kernel: A_ij = exp( -|x_i - x_j|^2 / h )
    // Fast decay, numerically very low rank.
    inline Matrix generate_gaussian(int N, double bandwidth, double diagonal_shift = 1e-12) {
        Matrix A(N, N);
        std::vector<double> x = generate_grid(N);
        
        for(int j = 0; j < N; ++j) {
            for(int i = 0; i < N; ++i) {
                double dist = x[i] - x[j];
                A(i, j) = std::exp(-(dist * dist) / (bandwidth * bandwidth));
            }
            // Regularization
            A(j, j) += diagonal_shift;
        }
        return A;
    }

    // 3. Coulomb / Laplace Kernel: A_ij = 1 / |x_i - x_j|
    // Slow decay, harder to compress than Gaussian, but valid for HODLR.
    inline Matrix generate_coulomb(int N, double diagonal_shift = 0.0) {
        Matrix A(N, N);
        std::vector<double> x = generate_grid(N);

        for(int j = 0; j < N; ++j) {
            for(int i = 0; i < N; ++i) {
                if (i == j) {
                    // Avoid singularity at 0 distance
                    // Set an arbitrary "self-interaction" value or use shift
                    A(i, j) = 0.0; 
                } else {
                    double dist = std::abs(x[i] - x[j]);
                    A(i, j) = 1.0 / dist;
                }
            }
             if (diagonal_shift != 0.0) {
                A(j, j) += diagonal_shift;
            }
        }
        return A;
    }

    // Wrapper to select kernel
    inline Matrix generate_matrix(Type type, int N) {
        switch (type) {
            case CAUCHY:
                // Shift to make it solvable
                return generate_cauchy(N, 1.0); 
            case GAUSSIAN:
                // Bandwidth 0.1
                return generate_gaussian(N, 0.1, 1e-6); 
            case COULOMB:
                // Shift to make it diagonally dominant/solvable
                return generate_coulomb(N, 10.0 * N); 
            default:
                return Matrix::Identity(N, N);
        }
    }
}

#endif // KERNEL_H