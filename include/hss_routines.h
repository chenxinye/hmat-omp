#ifndef HSS_ROUTINES_H
#define HSS_ROUTINES_H

#include "utils.h"
#include "compression.h"
#include <vector>

class HSSMatrix {
public:
    HSSNode* root;
    std::vector<HSSNode*> nodes;
    int N;
    double tolerance;
    int rsvd_rank;

    HSSMatrix(double tol = 1e-6) : root(nullptr), N(0), tolerance(tol), rsvd_rank(64) {}
    ~HSSMatrix() { for(auto n : nodes) delete n; }

    //  Serial Build (Using Fast RSVD) 
    void build_from_dense(const Matrix& A, int leaf_size) {
        N = A.rows();
        root = new HSSNode();
        nodes.push_back(root);
        build_recursive(root, A, 0, N, leaf_size);
    }

    void build_recursive(HSSNode* node, const Matrix& A, int start, int end, int leaf_size) {
        node->idx_start = start;
        node->idx_end = end;
        int len = end - start;

        if (len <= leaf_size) {
            node->is_leaf = true;
            node->D = A.block(start, start, len, len);
        } else {
            node->is_leaf = false;
            int mid = (start + end) / 2;
            
            node->left = new HSSNode(); 
            node->right = new HSSNode();
            
            // Recursive calls (Serial)
            build_recursive(node->left, A, start, mid, leaf_size);
            build_recursive(node->right, A, mid, end, leaf_size);
            
            // Off-Diagonal Compression (Same RSVD logic as OMP version)
            int len_l = mid - start;
            int len_r = end - mid;
            Matrix A12 = A.block(start, mid, len_l, len_r);
            Matrix A21 = A.block(mid, start, len_r, len_l);
            
            auto lr_12 = Compression::compress_rsvd(A12, tolerance, rsvd_rank);
            auto lr_21 = Compression::compress_rsvd(A21, tolerance, rsvd_rank);
            
            node->left->U = lr_12.U;
            node->right->V = lr_12.V;
            node->B_12 = lr_12.B;

            node->right->U = lr_21.U;
            node->left->V = lr_21.V;
            node->B_21 = lr_21.B;
        }
    }

    //  Serial Multiply 
    Matrix multiply(const Matrix& x) {
        return mult_recursive(root, x);
    }

    Matrix mult_recursive(HSSNode* node, const Matrix& x_global) {
        int start = node->idx_start;
        int len = node->idx_end - node->idx_start;
        
        if (node->is_leaf) {
            Matrix x_loc = x_global.block(start, 0, len, 1);
            return node->D * x_loc;
        } else {
            int mid = (node->idx_start + node->idx_end) / 2;
            int len_l = mid - node->idx_start;
            
            // Recursive calls (Serial)
            Matrix y_l = mult_recursive(node->left, x_global);
            Matrix y_r = mult_recursive(node->right, x_global);
            
            Matrix x_loc = x_global.block(start, 0, len, 1);
            Matrix x_l = x_loc.block(0, 0, len_l, 1);
            Matrix x_r = x_loc.block(len_l, 0, len - len_l, 1);
            
            Matrix t1 = node->right->V.transpose() * x_r;
            Matrix t2 = node->B_12 * t1; 
            Matrix term_l = node->left->U * t2; 
            
            Matrix t3 = node->left->V.transpose() * x_l;
            Matrix t4 = node->B_21 * t3;
            Matrix term_r = node->right->U * t4;

            for(int i=0; i<y_l.rows(); ++i) y_l(i,0) += term_l(i,0);
            for(int i=0; i<y_r.rows(); ++i) y_r(i,0) += term_r(i,0);
            
            Matrix y(len, 1);
            y.set_block(0, 0, y_l);
            y.set_block(len_l, 0, y_r);
            return y;
        }
    }

    //  Serial Solver (Woodbury) 
    Matrix solve(const Matrix& b) {
        return solve_woodbury(root, b);
    }

    Matrix solve_woodbury(HSSNode* node, const Matrix& b_loc) {
        int cols = b_loc.cols();
        if (node->is_leaf) {
            Matrix x = b_loc;
            Matrix D_inv = node->D; 
            D_inv.solve_lu(x); 
            return x;
        } else {
            int mid = (node->idx_start + node->idx_end) / 2;
            int len_l = mid - node->idx_start;
            int len_r = node->idx_end - mid;
            
            Matrix b1 = b_loc.block(0, 0, len_l, cols);
            Matrix b2 = b_loc.block(len_l, 0, len_r, cols);
            
            // Recursive calls (Serial)
            Matrix z1 = solve_woodbury(node->left, b1);
            Matrix z2 = solve_woodbury(node->right, b2);
            
            Matrix vt_z_1 = node->right->V.transpose() * z2; 
            Matrix vt_z_2 = node->left->V.transpose() * z1;  
            
            Matrix U1 = node->left->U;
            Matrix U2 = node->right->U;
            
            Matrix Q1 = solve_woodbury(node->left, U1);
            Matrix Q2 = solve_woodbury(node->right, U2);
            
            int r1 = node->B_12.rows();
            int r2 = node->B_21.rows();
            int R = r1 + r2;
            Matrix C(R, R); C.setZero();
            
            Matrix B12_inv = node->B_12.inverse();
            Matrix B21_inv = node->B_21.inverse();
            C.set_block(0, 0, B12_inv);
            C.set_block(r1, r1, B21_inv);
            
            Matrix TR = node->right->V.transpose() * Q2;
            C.set_block(0, r1, TR);
            
            Matrix BL = node->left->V.transpose() * Q1;
            C.set_block(r1, 0, BL);
            
            Matrix rhs_cap(R, cols);
            rhs_cap.set_block(0, 0, vt_z_1);
            rhs_cap.set_block(r1, 0, vt_z_2);
            
            C.solve_lu(rhs_cap); 
            
            Matrix g1 = rhs_cap.block(0, 0, r1, cols);
            Matrix g2 = rhs_cap.block(r1, 0, r2, cols);
            
            Matrix corr1 = Q1 * g1;
            Matrix corr2 = Q2 * g2;
            
            Matrix x(b_loc.rows(), cols);
            for(int j=0; j<cols; ++j) {
                for(int i=0; i<len_l; ++i) x(i,j) = z1(i,j) - corr1(i,j);
                for(int i=0; i<len_r; ++i) x(len_l+i,j) = z2(i,j) - corr2(i,j);
            }
            return x;
        }
    }
};

#endif