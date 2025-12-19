#ifndef HSS_ROUTINES_H
#define HSS_ROUTINES_H

#include "utils.h"
#include "compression.h"
#include <vector>
#include <cmath>
#include <algorithm>

class HSSMatrix {
public:
    HSSNode* root;
    std::vector<HSSNode*> nodes;
    int N;
    double tolerance;
    int rsvd_rank;

    HSSMatrix(double tol = 1e-9) : root(nullptr), N(0), tolerance(tol), rsvd_rank(64) {}
    ~HSSMatrix() { for(auto n : nodes) delete n; }

    // --- Helper: Reconstruct Dense U Basis ---
    // U_parent = [U_L 0; 0 U_R] * R
    Matrix reconstruct_U(HSSNode* node) {
        if (node->is_leaf) return node->U;
        Matrix UL = reconstruct_U(node->left);
        Matrix UR = reconstruct_U(node->right);
        int r1 = UL.rows(), c1 = UL.cols();
        int r2 = UR.rows(), c2 = UR.cols();
        Matrix U_big = Matrix::Zero(r1 + r2, c1 + c2);
        U_big.set_block(0, 0, UL);
        U_big.set_block(r1, c1, UR);
        return U_big * node->R;
    }

    Matrix reconstruct_V(HSSNode* node) {
        if (node->is_leaf) return node->V;
        Matrix VL = reconstruct_V(node->left);
        Matrix VR = reconstruct_V(node->right);
        int r1 = VL.rows(), c1 = VL.cols();
        int r2 = VR.rows(), c2 = VR.cols();
        Matrix V_big = Matrix::Zero(r1 + r2, c1 + c2);
        V_big.set_block(0, 0, VL);
        V_big.set_block(r1, c1, VR);
        return V_big * node->W;
    }

    // ========================================================
    // 1. Build
    // ========================================================
    void build_from_dense(const Matrix& A, int leaf_size) {
        N = A.rows();
        nodes.clear();
        root = new HSSNode(); root->id = 0; nodes.push_back(root);
        build_topo(root, 0, N, leaf_size);
        
        // 必须自底向上构建，因为父节点依赖子节点的 U/V
        for (int i = nodes.size() - 1; i >= 0; --i) {
            HSSNode* node = nodes[i];
            int start = node->idx_start;
            int len = node->idx_end - node->idx_start;

            if (node->is_leaf) {
                node->D = A.block(start, start, len, len);
                
                // Compress Rows
                Matrix RowBlock = A.block(start, 0, len, N);
                for(int r=0; r<len; ++r) for(int c=start; c<node->idx_end; ++c) RowBlock(r, c) = 0.0;
                auto lr_U = Compression::compress_rsvd(RowBlock, tolerance, rsvd_rank);
                node->U = lr_U.U; 
                node->rank_U = node->U.cols();

                // Compress Cols
                Matrix ColBlockT = A.block(0, start, N, len).transpose();
                for(int r=0; r<len; ++r) for(int c=start; c<node->idx_end; ++c) ColBlockT(r, c) = 0.0;
                auto lr_V = Compression::compress_rsvd(ColBlockT, tolerance, rsvd_rank);
                node->V = lr_V.U; 
                node->rank_V = node->V.cols();
            } else {
                HSSNode* lc = node->left;
                HSSNode* rc = node->right;
                
                node->rank_U = lc->rank_U + rc->rank_U;
                node->rank_V = lc->rank_V + rc->rank_V;

                // Identity Transfer
                if (node->rank_U > 0) node->R = Matrix::Identity(node->rank_U, node->rank_U);
                else node->R = Matrix::Zero(0, 0);

                if (node->rank_V > 0) node->W = Matrix::Identity(node->rank_V, node->rank_V);
                else node->W = Matrix::Zero(0, 0);
                
                if (node->rank_U > 0 && node->rank_V > 0) {
                    int mid = lc->idx_end;
                    Matrix A12 = A.block(start, mid, mid-start, node->idx_end-mid); 
                    Matrix A21 = A.block(mid, start, node->idx_end-mid, mid-start); 
                    
                    Matrix UL = reconstruct_U(lc);
                    Matrix VR = reconstruct_V(rc);
                    Matrix UR = reconstruct_U(rc);
                    Matrix VL = reconstruct_V(lc);

                    Matrix B12 = UL.transpose() * A12 * VR;
                    Matrix B21 = UR.transpose() * A21 * VL;
                    
                    node->B = Matrix::Zero(node->rank_U, node->rank_V);
                    node->B.set_block(0, lc->rank_V, B12);
                    node->B.set_block(lc->rank_U, 0, B21);
                } else {
                    node->B = Matrix::Zero(0, 0);
                }
            }
        }
    }

    // ========================================================
    // 2. Multiply
    // ========================================================
    Matrix multiply(const Matrix& x) {
        int num_nodes = nodes.size();
        std::vector<Matrix> g(num_nodes);
        std::vector<Matrix> f(num_nodes);

        // Up-Sweep
        for (int i = num_nodes - 1; i >= 0; --i) {
            HSSNode* node = nodes[i];
            if (node->is_leaf) {
                Matrix x_loc = x.block(node->idx_start, 0, node->idx_end - node->idx_start, 1);
                if (node->rank_V > 0) g[i] = node->V.transpose() * x_loc;
                else g[i] = Matrix::Zero(0, 1);
            } else {
                if (node->rank_V == 0) {
                     g[i] = Matrix::Zero(0, 1);
                } else {
                    Matrix g_l = g[node->left->id];
                    Matrix g_r = g[node->right->id];
                    Matrix g_stack = Matrix::Zero(g_l.rows() + g_r.rows(), 1);
                    g_stack.set_block(0, 0, g_l);
                    g_stack.set_block(g_l.rows(), 0, g_r);
                    g[i] = node->W.transpose() * g_stack;
                }
            }
        }

        // Down-Sweep
        if (root->rank_U > 0) f[0] = Matrix::Zero(root->rank_U, 1);
        else f[0] = Matrix::Zero(0, 1);

        for (int i = 0; i < num_nodes; ++i) {
            HSSNode* node = nodes[i];
            if (!node->is_leaf) {
                if (node->rank_U == 0) {
                     f[node->left->id] = Matrix::Zero(node->left->rank_U, 1);
                     f[node->right->id] = Matrix::Zero(node->right->rank_U, 1);
                     continue;
                }

                Matrix g_l = g[node->left->id];
                Matrix g_r = g[node->right->id];
                Matrix g_stack = Matrix::Zero(g_l.rows() + g_r.rows(), 1);
                g_stack.set_block(0, 0, g_l);
                g_stack.set_block(g_l.rows(), 0, g_r);
                
                Matrix interact = node->B * g_stack;
                Matrix f_pass = node->R * f[i]; 
                Matrix f_total = f_pass + interact;
                
                int rl = node->left->rank_U;
                int rr = node->right->rank_U;
                
                f[node->left->id] = f_total.block(0, 0, rl, 1);
                f[node->right->id] = f_total.block(rl, 0, rr, 1);
            }
        }

        // Local
        Matrix y(N, 1);
        for (int i = 0; i < num_nodes; ++i) {
            HSSNode* node = nodes[i];
            if (node->is_leaf) {
                int start = node->idx_start;
                int len = node->idx_end - node->idx_start;
                Matrix x_loc = x.block(start, 0, len, 1);
                Matrix y_loc = node->D * x_loc;
                if (node->rank_U > 0) {
                    Matrix uf = node->U * f[i];
                    for(int k=0; k<len; ++k) y_loc(k,0) += uf(k,0);
                }
                y.set_block(start, 0, y_loc);
            }
        }
        return y;
    }

    // ========================================================
    // 3. Direct Solver (Stable Formulation)
    // ========================================================
    Matrix solve(const Matrix& b) {
        return solve_multi_rhs(root, b);
    }
    
    // Solves Ax = b using Stable Identity:
    // (D + U B V^T)^-1 = D^-1 - D^-1 U B (I + V^T D^-1 U B)^-1 V^T D^-1
    // Let Q = D^-1 U, z = D^-1 b
    // x = z - Q * B * (I + V^T Q B)^-1 * V^T z
    Matrix solve_recursive(HSSNode* node, const Matrix& b_loc) {
        if (node->is_leaf) {
            Matrix x = b_loc;
            Matrix D_inv = node->D;
            D_inv.solve_lu(x);
            return x;
        } 
        
        // 1. Recursive solve for Diagonal blocks (z = D^-1 * b)
        int mid = (node->idx_start + node->idx_end) / 2;
        int len_l = mid - node->idx_start;
        int len_r = node->idx_end - mid;
        
        Matrix b1 = b_loc.block(0,0, len_l, 1);
        Matrix b2 = b_loc.block(len_l,0, len_r, 1);
        
        Matrix z1 = solve_recursive(node->left, b1);
        Matrix z2 = solve_recursive(node->right, b2);

        if (node->rank_U == 0 || node->rank_V == 0) {
            Matrix x(b_loc.rows(), 1);
            x.set_block(0, 0, z1);
            x.set_block(len_l, 0, z2);
            return x;
        }
        
        // 2. Compute Q = D^-1 * U
        Matrix U_L_dense = reconstruct_U(node->left);
        Matrix U_R_dense = reconstruct_U(node->right);
        
        Matrix Q1 = solve_multi_rhs(node->left, U_L_dense);
        Matrix Q2 = solve_multi_rhs(node->right, U_R_dense);
        
        // 3. Form Core System: (I + K * B) * gamma = V^T * z
        // where K = V^T * Q
        Matrix V_L_dense = reconstruct_V(node->left);
        Matrix V_R_dense = reconstruct_V(node->right);
        
        Matrix VTQ1 = V_L_dense.transpose() * Q1;
        Matrix VTQ2 = V_R_dense.transpose() * Q2;
        
        Matrix K = Matrix::Zero(node->rank_V, node->rank_U);
        K.set_block(0, 0, VTQ1);
        // 注意：HSS定义中，V_R 对应 B 的右上块还是右下块取决于 B 的结构
        // 此时 V_big = diag(VL, VR) * W. K = diag(VTQ1, VTQ2).
        K.set_block(node->left->rank_V, node->left->rank_U, VTQ2);
        
        // M = I + K * B
        Matrix M = K * node->B; 
        for(int i=0; i<M.rows(); ++i) M(i,i) += 1.0;
        
        // RHS_gamma = V^T * z
        Matrix VTz_1 = V_L_dense.transpose() * z1;
        Matrix VTz_2 = V_R_dense.transpose() * z2;
        Matrix VTz = Matrix::Zero(node->rank_V, 1);
        VTz.set_block(0, 0, VTz_1);
        VTz.set_block(node->left->rank_V, 0, VTz_2);
        
        // Solve M * gamma_tilde = VTz
        M.solve_lu(VTz); // VTz is now gamma_tilde
        
        // 4. Update x = z - Q * (B * gamma_tilde)
        Matrix B_gamma = node->B * VTz;
        Matrix gam1 = B_gamma.block(0, 0, node->left->rank_U, 1);
        Matrix gam2 = B_gamma.block(node->left->rank_U, 0, node->right->rank_U, 1); 
        
        Matrix corr1 = Q1 * gam1;
        Matrix corr2 = Q2 * gam2;
        
        Matrix x(b_loc.rows(), 1);
        for(int i=0; i<len_l; ++i) x(i,0) = z1(i,0) - corr1(i,0);
        for(int i=0; i<len_r; ++i) x(len_l+i,0) = z2(i,0) - corr2(i,0);
        
        return x;
    }
    
    Matrix solve_multi_rhs(HSSNode* node, const Matrix& B_mat) {
        int cols = B_mat.cols();
        Matrix Res(B_mat.rows(), cols);
        for(int j=0; j<cols; ++j) {
             Matrix col = B_mat.block(0, j, B_mat.rows(), 1);
             Matrix sol = solve_recursive(node, col);
             Res.set_block(0, j, sol);
        }
        return Res;
    }

private:
    void build_topo(HSSNode* node, int start, int end, int leaf_size) {
        node->idx_start = start;
        node->idx_end = end;
        if (end - start <= leaf_size) {
            node->is_leaf = true;
        } else {
            int mid = (start + end) / 2;
            node->left = new HSSNode(); node->left->id = nodes.size(); nodes.push_back(node->left);
            node->right = new HSSNode(); node->right->id = nodes.size(); nodes.push_back(node->right);
            node->left->parent = node; node->right->parent = node;
            build_topo(node->left, start, mid, leaf_size);
            build_topo(node->right, mid, end, leaf_size);
        }
    }
};

#endif