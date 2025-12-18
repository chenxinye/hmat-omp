#ifndef MATRIX_H
#define MATRIX_H

#include <vector>
#include <iostream>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <cassert>

// LAPACK / BLAS C Interface Definitions 
extern "C" {
    // BLAS: Matrix Multiplication
    void dgemm_(const char* transa, const char* transb, const int* m, const int* n, const int* k,
                const double* alpha, const double* A, const int* lda,
                const double* B, const int* ldb, const double* beta,
                double* C, const int* ldc);

    // LAPACK: SVD
    void dgesvd_(const char* jobu, const char* jobvt, const int* m, const int* n, double* a, const int* lda,
                 double* s, double* u, const int* ldu, double* vt, const int* ldvt,
                 double* work, const int* lwork, int* info);
    
    // LAPACK: LU Decomposition (dgetrf) and Solve (dgetrs)
    void dgetrf_(const int* m, const int* n, double* a, const int* lda, int* ipiv, int* info);
    void dgetrs_(const char* trans, const int* n, const int* nrhs, const double* a, const int* lda, const int* ipiv, double* b, const int* ldb, int* info);

    // LAPACK: Cholesky Decomposition (dpotrf) and Solve (dpotrs)
    void dpotrf_(const char* uplo, const int* n, double* a, const int* lda, int* info);
    void dpotrs_(const char* uplo, const int* n, const int* nrhs, const double* a, const int* lda, double* b, const int* ldb, int* info);
}

// Lightweight Matrix Class (Column-Major) 
class Matrix {
public:
    int rows_, cols_;
    std::vector<double> data_;

    Matrix() : rows_(0), cols_(0) {}
    Matrix(int r, int c) : rows_(r), cols_(c), data_(r * c, 0.0) {}

    // Accessors (0-based)
    double& operator()(int i, int j) { return data_[j * rows_ + i]; }
    const double& operator()(int i, int j) const { return data_[j * rows_ + i]; }

    int rows() const { return rows_; }
    int cols() const { return cols_; }
    double* data() { return data_.data(); }
    const double* data() const { return data_.data(); }

    void setZero() { std::fill(data_.begin(), data_.end(), 0.0); }
    
    // Static helpers
    static Matrix Random(int r, int c) {
        Matrix M(r, c);
        for(auto& val : M.data_) val = (double)rand() / RAND_MAX;
        return M;
    }

    static Matrix Identity(int r, int c) {
        Matrix M(r, c);
        M.setZero();
        for(int i=0; i<std::min(r,c); ++i) M(i,i) = 1.0;
        return M;
    }

    // Extract sub-block
    Matrix block(int r_start, int c_start, int r_len, int c_len) const {
        Matrix res(r_len, c_len);
        for(int j=0; j<c_len; ++j) {
            for(int i=0; i<r_len; ++i) {
                res(i, j) = (*this)(r_start + i, c_start + j);
            }
        }
        return res;
    }

    // Set sub-block
    void set_block(int r_start, int c_start, const Matrix& B) {
        for(int j=0; j<B.cols(); ++j) {
            for(int i=0; i<B.rows(); ++i) {
                if(r_start+i < rows_ && c_start+j < cols_)
                    (*this)(r_start + i, c_start + j) = B(i, j);
            }
        }
    }

    Matrix transpose() const {
        Matrix T(cols_, rows_);
        for(int j=0; j<cols_; ++j)
            for(int i=0; i<rows_; ++i)
                T(j, i) = (*this)(i, j);
        return T;
    }

    // GEMV / GEMM: Matrix Multiplication
    Matrix operator*(const Matrix& B) const {
        assert(cols_ == B.rows_);
        Matrix C(rows_, B.cols_);
        char trans = 'N';
        double alpha = 1.0, beta = 0.0;
        int m = rows_, n = B.cols_, k = cols_;
        int lda = rows_, ldb = B.rows_, ldc = rows_;

        dgemm_(&trans, &trans, &m, &n, &k, &alpha, data_.data(), &lda, 
               B.data(), &ldb, &beta, C.data(), &ldc);
        return C;
    }

    // Linear Solvers Wrappers 

    // LU Solve (Ax = b), b is overwritten with x
    void solve_lu(Matrix& b) {
        assert(rows_ == cols_);
        assert(rows_ == b.rows_);
        int n = rows_;
        int nrhs = b.cols_;
        int lda = rows_, ldb = rows_, info;
        std::vector<int> ipiv(n);
        char trans = 'N';

        dgetrf_(&n, &n, data_.data(), &lda, ipiv.data(), &info);
        if(info != 0) std::cerr << "LU Factorization failed! Info: " << info << std::endl;

        dgetrs_(&trans, &n, &nrhs, data_.data(), &lda, ipiv.data(), b.data(), &ldb, &info);
    }

    // Cholesky Solve (Ax = b, SPD), b is overwritten with x
    void solve_chol(Matrix& b) {
        assert(rows_ == cols_);
        assert(rows_ == b.rows_);
        int n = rows_;
        int nrhs = b.cols_;
        int lda = rows_, ldb = rows_, info;
        char uplo = 'L'; 

        dpotrf_(&uplo, &n, data_.data(), &lda, &info);
        if(info != 0) std::cerr << "Cholesky Factorization failed! Info: " << info << std::endl;

        dpotrs_(&uplo, &n, &nrhs, data_.data(), &lda, b.data(), &ldb, &info);
    }
};

#endif // MATRIX_H