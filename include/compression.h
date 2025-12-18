#ifndef COMPRESSION_H
#define COMPRESSION_H

#include "matrix.h"
#include <random>
#include <algorithm>

namespace Compression {

    struct LowRank {
        Matrix U; // Orthonormal Columns
        Matrix V; // Orthonormal Columns
        Matrix B; // Core Scaling Matrix (Small)
        int rank;
    };

    // Standard SVD: A ~ U * S * Vt -> We return U, V, and B=diag(S)
    inline LowRank compress_svd(const Matrix& A, double tol) {
        int m = A.rows();
        int n = A.cols();
        if (m == 0 || n == 0) return {Matrix(0,0), Matrix(0,0), Matrix(0,0), 0};
        
        int min_dim = std::min(m, n);
        
        char jobu = 'S', jobvt = 'S';
        Matrix U(m, min_dim), Vt(min_dim, n);
        std::vector<double> S(min_dim);
        Matrix A_copy = A;
        
        int lda = m, ldu = m, ldvt = min_dim, info, lwork = -1;
        double work_query;
        
        dgesvd_(&jobu, &jobvt, &m, &n, A_copy.data(), &lda, S.data(), 
                U.data(), &ldu, Vt.data(), &ldvt, &work_query, &lwork, &info);
        
        lwork = (int)work_query;
        std::vector<double> work(lwork);
        dgesvd_(&jobu, &jobvt, &m, &n, A_copy.data(), &lda, S.data(), 
                U.data(), &ldu, Vt.data(), &ldvt, work.data(), &lwork, &info);

        // Determine Rank
        int rank = 0;
        if (min_dim > 0 && S[0] > 1e-15) {
            for(int i=0; i<min_dim; ++i) if(S[i] > tol * S[0]) rank++;
        }
        if(rank == 0) rank = 1;

        // Extract U (m x r), V (n x r), B (r x r)
        Matrix U_ret(m, rank);
        Matrix V_ret(n, rank);
        Matrix B_ret = Matrix::Identity(rank, rank);
        B_ret.setZero();

        for(int j=0; j<rank; ++j) {
            // U is already orthonormal from LAPACK
            for(int i=0; i<m; ++i) U_ret(i,j) = U(i,j);
            // Vt rows are orthonormal. V_ret needs columns.
            for(int i=0; i<n; ++i) V_ret(i,j) = Vt(j,i); 
            // Store Sigma in B
            B_ret(j,j) = S[j];
        }

        return {U_ret, V_ret, B_ret, rank};
    }

    // Randomized SVD: Returns U, V (Ortho) and B (Core)
    inline LowRank compress_rsvd(const Matrix& A, double tol, int rank_guess) {
        int m = A.rows();
        int n = A.cols();
        
        // Fallback for small blocks
        if (m < 256 || n < 256) return compress_svd(A, tol);

        int k = rank_guess + 10; 
        if (k > std::min(m,n)) k = std::min(m,n);

        // 1. Sample Range: Y = A * Omega
        Matrix Omega = Matrix::Random(n, k);
        Matrix Y = A * Omega;

        // 2. Orthogonalize Y -> Q
        // We use SVD of Y to get Q (U of Y)
        LowRank qr = compress_svd(Y, -1.0); 
        Matrix Q = qr.U; // Orthonormal basis for Range(A)

        // 3. Project: B_small = Q^T * A
        Matrix B_small = Q.transpose() * A;

        // 4. SVD of B_small
        LowRank svd_b = compress_svd(B_small, tol);

        // 5. Recover: A ~ Q * (U_b * S_b * V_b^T)
        // Final U = Q * U_b (Product of ortho is ortho)
        Matrix U_final = Q * svd_b.U;
        Matrix V_final = svd_b.V;
        Matrix B_final = svd_b.B;

        return {U_final, V_final, B_final, svd_b.rank};
    }
}

#endif