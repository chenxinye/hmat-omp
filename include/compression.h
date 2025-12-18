#ifndef COMPRESSION_H
#define COMPRESSION_H

#include "matrix.h"

namespace Compression {

    struct LowRank {
        Matrix U;
        Matrix V; // Approximation = U * V^T
        int rank;
    };

    // SVD Compression using LAPACK dgesvd
    // Returns {U, V} such that A approx U * V^T
    inline LowRank compress_svd(const Matrix& A, double tol) {
        int m = A.rows();
        int n = A.cols();
        
        if (m == 0 || n == 0) return {Matrix(m,0), Matrix(0,n), 0};

        int min_dim = std::min(m, n);
        
        // LAPACK Workspace
        char jobu = 'S'; 
        char jobvt = 'S'; 
        
        Matrix U(m, min_dim);
        Matrix Vt(min_dim, n);
        std::vector<double> S(min_dim);
        
        Matrix A_copy = A; // Copy because LAPACK modifies input

        int lda = m, ldu = m, ldvt = min_dim;
        int info;
        int lwork = -1;
        double work_query;
        
        // 1. Workspace Query
        dgesvd_(&jobu, &jobvt, &m, &n, A_copy.data(), &lda, S.data(), 
                U.data(), &ldu, Vt.data(), &ldvt, &work_query, &lwork, &info);
        
        lwork = (int)work_query;
        std::vector<double> work(lwork);

        // 2. Compute SVD
        dgesvd_(&jobu, &jobvt, &m, &n, A_copy.data(), &lda, S.data(), 
                U.data(), &ldu, Vt.data(), &ldvt, work.data(), &lwork, &info);

        // 3. Determine Rank based on tolerance
        int rank = 0;
        if (min_dim > 0) {
            double max_s = S[0];
            for(int i=0; i<min_dim; ++i) {
                if(S[i] > tol * max_s) rank++;
            }
        }
        if(rank == 0 && min_dim > 0) rank = 1; 

        // 4. Form U and V
        // SVD gives A = U * Sigma * Vt
        // We want A = (U*Sigma) * (Vt^T)^T -> so our "V" is Vt^T.
        // Wait, standard definition is A = U * V^T. 
        // LAPACK returns Vt directly. So "V" in our struct is Vt.transpose().
        // BUT, for efficiency, let's store V such that A = U * V^T.
        // Since LAPACK gives Vt, and Vt rows are eigenvectors.
        // Let's adhere to: result.V should have size (n x rank).
        
        Matrix U_ret(m, rank);
        Matrix V_ret(n, rank); 

        for(int j=0; j<rank; ++j) {
            // Scale U by singular values
            for(int i=0; i<m; ++i) U_ret(i,j) = U(i,j) * S[j];
            
            // Transpose Vt to get V
            for(int i=0; i<n; ++i) V_ret(i,j) = Vt(j,i); 
        }

        return {U_ret, V_ret, rank};
    }
}

#endif // COMPRESSION_H