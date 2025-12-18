#ifndef MATRIX_H
#define MATRIX_H

#include <vector>
#include <iostream>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <cassert>

//  LAPACK / BLAS C Interface Definitions 
extern "C" {
    // BLAS
    void dgemm_(const char* transa, const char* transb, const int* m, const int* n, const int* k,
                const double* alpha, const double* A, const int* lda,
                const double* B, const int* ldb, const double* beta,
                double* C, const int* ldc);

    // LAPACK SVD / QR
    void dgesvd_(const char* jobu, const char* jobvt, const int* m, const int* n, double* a, const int* lda,
                 double* s, double* u, const int* ldu, double* vt, const int* ldvt,
                 double* work, const int* lwork, int* info);
    
    // LAPACK LU
    void dgetrf_(const int* m, const int* n, double* a, const int* lda, int* ipiv, int* info);
    void dgetrs_(const char* trans, const int* n, const int* nrhs, const double* a, const int* lda, const int* ipiv, double* b, const int* ldb, int* info);
    
    // LAPACK Inverse (using LU)
    void dgetri_(const int* n, double* a, const int* lda, const int* ipiv, double* work, const int* lwork, int* info);

    // LAPACK Cholesky
    void dpotrf_(const char* uplo, const int* n, double* a, const int* lda, int* info);
    void dpotrs_(const char* uplo, const int* n, const int* nrhs, const double* a, const int* lda, double* b, const int* ldb, int* info);
}

//  Lightweight Matrix Class 
class Matrix {
public:
    int rows_, cols_;
    std::vector<double> data_;

    Matrix() : rows_(0), cols_(0) {}
    Matrix(int r, int c) : rows_(r), cols_(c), data_(r * c, 0.0) {}

    // Accessors
    double& operator()(int i, int j) { return data_[j * rows_ + i]; }
    const double& operator()(int i, int j) const { return data_[j * rows_ + i]; }

    int rows() const { return rows_; }
    int cols() const { return cols_; }
    double* data() { return data_.data(); }
    const double* data() const { return data_.data(); }

    void setZero() { std::fill(data_.begin(), data_.end(), 0.0); }
    
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

    Matrix block(int r_start, int c_start, int r_len, int c_len) const {
        Matrix res(r_len, c_len);
        for(int j=0; j<c_len; ++j) {
            for(int i=0; i<r_len; ++i) {
                res(i, j) = (*this)(r_start + i, c_start + j);
            }
        }
        return res;
    }

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

    void solve_lu(Matrix& b) {
        assert(rows_ == cols_);
        assert(rows_ == b.rows_);
        int n = rows_;
        int nrhs = b.cols_;
        int lda = rows_, ldb = rows_, info;
        std::vector<int> ipiv(n);
        char trans = 'N';

        dgetrf_(&n, &n, data_.data(), &lda, ipiv.data(), &info);
        dgetrs_(&trans, &n, &nrhs, data_.data(), &lda, ipiv.data(), b.data(), &ldb, &info);
    }

    //  Inverse Method 
    Matrix inverse() const {
        assert(rows_ == cols_);
        int n = rows_;
        Matrix Inv = *this; // Copy
        int lda = n, info;
        std::vector<int> ipiv(n);
        
        // 1. LU Factorization
        dgetrf_(&n, &n, Inv.data(), &lda, ipiv.data(), &info);
        
        // 2. Invert using LU factors
        // Query workspace
        double work_query;
        int lwork = -1;
        dgetri_(&n, Inv.data(), &lda, ipiv.data(), &work_query, &lwork, &info);
        
        lwork = (int)work_query;
        std::vector<double> work(lwork);
        dgetri_(&n, Inv.data(), &lda, ipiv.data(), work.data(), &lwork, &info);
        
        return Inv;
    }
};

#endif