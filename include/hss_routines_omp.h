#ifndef HSS_ROUTINES_OMP_H
#define HSS_ROUTINES_OMP_H

#include "utils.h"
#include "compression.h"
#include <vector>
#include <cmath>
#include <algorithm>
#include <omp.h>

class HSSMatrixOMP {
public:
    HSSNode* root;
    std::vector<HSSNode*> nodes;
    int N;
    double tolerance;
    int rsvd_rank;

    HSSMatrixOMP(double tol = 1e-9) : root(nullptr), N(0), tolerance(tol), rsvd_rank(64) {}
    ~HSSMatrixOMP() { for(auto n : nodes) delete n; }

    // --- Helper: Task-based Reconstruction ---
    Matrix reconstruct_U(HSSNode* node) {
        if (node->is_leaf) return node->U;
        
        Matrix UL, UR;
        #pragma omp task shared(UL)
        UL = reconstruct_U(node->left);
        #pragma omp task shared(UR)
        UR = reconstruct_U(node->right);
        #pragma omp taskwait

        int r1 = UL.rows(), c1 = UL.cols();
        int r2 = UR.rows(), c2 = UR.cols();
        Matrix U_big = Matrix::Zero(r1 + r2, c1 + c2);
        U_big.set_block(0, 0, UL);
        U_big.set_block(r1, c1, UR);
        return U_big * node->R;
    }

    Matrix reconstruct_V(HSSNode* node) {
        if (node->is_leaf) return node->V;
        
        Matrix VL, VR;
        #pragma omp task shared(VL)
        VL = reconstruct_V(node->left);
        #pragma omp task shared(VR)
        VR = reconstruct_V(node->right);
        #pragma omp taskwait

        int r1 = VL.rows(), c1 = VL.cols();
        int r2 = VR.rows(), c2 = VR.cols();
        Matrix V_big = Matrix::Zero(r1 + r2, c1 + c2);
        V_big.set_block(0, 0, VL);
        V_big.set_block(r1, c1, VR);
        return V_big * node->W;
    }

    // ========================================================
    // 1. Build Phase (Parallelized)
    // ========================================================
    void build_from_dense(const Matrix& A, int leaf_size) {
        N = A.rows();
        nodes.clear();
        root = new HSSNode(); root->id = 0; nodes.push_back(root);
        build_topo(root, 0, N, leaf_size);

        // Step 1: Compress Leaves Parallel
        #pragma omp parallel for schedule(dynamic)
        for (size_t i = 0; i < nodes.size(); ++i) {
            HSSNode* node = nodes[i];
            if (node->is_leaf) {
                int start = node->idx_start;
                int len = node->idx_end - node->idx_start;
                node->D = A.block(start, start, len, len);

                // Row Basis
                Matrix RowBlock = A.block(start, 0, len, N);
                for (int r = 0; r < len; ++r)
                    for (int c = start; c < node->idx_end; ++c)
                        RowBlock(r, c) = 0.0;
                auto lr_U = Compression::compress_rsvd(RowBlock, tolerance, rsvd_rank);
                node->U = std::move(lr_U.U);
                node->rank_U = node->U.cols();

                // Col Basis
                Matrix ColBlockT = A.block(0, start, N, len).transpose();
                for (int r = 0; r < len; ++r)
                    for (int c = start; c < node->idx_end; ++c)
                        ColBlockT(r, c) = 0.0;
                auto lr_V = Compression::compress_rsvd(ColBlockT, tolerance, rsvd_rank);
                node->V = std::move(lr_V.U);
                node->rank_V = node->V.cols();
            }
        }

        // Step 2: Build Internal Nodes (Bottom-Up)
        // Use a parallel region but synchronize levels or use tasks. 
        // Simple reversed loop with internal tasking is effective enough.
        #pragma omp parallel
        {
            #pragma omp single
            {
                for (int i = nodes.size() - 1; i >= 0; --i) {
                    HSSNode* node = nodes[i];
                    if (!node->is_leaf) {
                        HSSNode* lc = node->left;
                        HSSNode* rc = node->right;
                        
                        node->rank_U = lc->rank_U + rc->rank_U;
                        node->rank_V = lc->rank_V + rc->rank_V;

                        if (node->rank_U == 0 || node->rank_V == 0) {
                            node->R = Matrix::Zero(0, 0);
                            node->W = Matrix::Zero(0, 0);
                            node->B = Matrix::Zero(0, 0);
                        } else {
                            node->R = Matrix::Identity(node->rank_U, node->rank_U);
                            node->W = Matrix::Identity(node->rank_V, node->rank_V);
                            
                            int start = node->idx_start;
                            int mid = lc->idx_end;
                            Matrix A12 = A.block(start, mid, mid - start, node->idx_end - mid);
                            Matrix A21 = A.block(mid, start, node->idx_end - mid, mid - start);
                            
                            Matrix UL, VR, UR, VL;
                            #pragma omp task shared(UL)
                            UL = reconstruct_U(lc);
                            #pragma omp task shared(VR)
                            VR = reconstruct_V(rc);
                            #pragma omp task shared(UR)
                            UR = reconstruct_U(rc);
                            #pragma omp task shared(VL)
                            VL = reconstruct_V(lc);
                            #pragma omp taskwait

                            // B12 = U_L^T * A12 * V_R
                            Matrix B12 = UL.transpose() * A12 * VR;
                            Matrix B21 = UR.transpose() * A21 * VL;
                            
                            node->B = Matrix::Zero(node->rank_U, node->rank_V);
                            node->B.set_block(0, lc->rank_V, B12);
                            node->B.set_block(lc->rank_U, 0, B21);
                        }
                    }
                }
            }
        }
    }

    // ========================================================
    // 2. Matrix-Vector Multiply
    // ========================================================
    Matrix multiply(const Matrix& x) {
        int num_nodes = nodes.size();
        std::vector<Matrix> g(num_nodes);
        std::vector<Matrix> f(num_nodes);

        // Up-Sweep
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < num_nodes; ++i) {
            HSSNode* node = nodes[i];
            if (node->is_leaf) {
                Matrix x_loc = x.block(node->idx_start, 0, node->idx_end - node->idx_start, 1);
                if (node->rank_V > 0) g[node->id] = node->V.transpose() * x_loc;
                else g[node->id] = Matrix::Zero(0, 1);
            }
        }

        // Sequential Up (Dependencies)
        for (int i = num_nodes - 1; i >= 0; --i) {
            HSSNode* node = nodes[i];
            if (!node->is_leaf) {
                if (node->rank_V == 0) {
                    g[node->id] = Matrix::Zero(0, 1);
                } else {
                    Matrix g_l = g[node->left->id];
                    Matrix g_r = g[node->right->id];
                    Matrix g_stack = Matrix::Zero(g_l.rows() + g_r.rows(), 1);
                    g_stack.set_block(0, 0, g_l);
                    g_stack.set_block(g_l.rows(), 0, g_r);
                    g[node->id] = node->W.transpose() * g_stack;
                }
            }
        }

        // Down-Sweep
        f[0] = (root->rank_U > 0) ? Matrix::Zero(root->rank_U, 1) : Matrix::Zero(0, 1);
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
                Matrix f_pass = node->R * f[node->id];
                Matrix f_total = f_pass + interact;
                
                int rl = node->left->rank_U;
                f[node->left->id] = f_total.block(0, 0, rl, 1);
                f[node->right->id] = f_total.block(rl, 0, f_total.rows() - rl, 1);
            }
        }

        Matrix y(N, 1);
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < num_nodes; ++i) {
            HSSNode* node = nodes[i];
            if (node->is_leaf) {
                int start = node->idx_start;
                int len = node->idx_end - node->idx_start;
                Matrix x_loc = x.block(start, 0, len, 1);
                Matrix y_loc = node->D * x_loc;
                if (node->rank_U > 0) {
                    Matrix uf = node->U * f[node->id];
                    for (int k = 0; k < len; ++k) y_loc(k, 0) += uf(k, 0);
                }
                y.set_block(start, 0, y_loc);
            }
        }
        return y;
    }

    // ========================================================
    // 3. Solver (Task Parallel + Stable Formula)
    // ========================================================
    Matrix solve(const Matrix& b) {
        Matrix res;
        #pragma omp parallel
        {
            #pragma omp single
            {
                res = solve_recursive(root, b);
            }
        }
        return res;
    }

    Matrix solve_recursive(HSSNode* node, const Matrix& b_loc) {
        if (node->is_leaf) {
            Matrix x = b_loc;
            Matrix D_inv = node->D;
            D_inv.solve_lu(x);
            return x;
        }

        if (node->rank_U == 0 || node->rank_V == 0) {
            // Just recurse, no correction needed
            int mid = (node->idx_start + node->idx_end) / 2;
            int len_l = mid - node->idx_start;
            int len_r = node->idx_end - mid;
            
            Matrix x1, x2;
            #pragma omp task shared(x1)
            x1 = solve_recursive(node->left, b_loc.block(0, 0, len_l, 1));
            #pragma omp task shared(x2)
            x2 = solve_recursive(node->right, b_loc.block(len_l, 0, len_r, 1));
            #pragma omp taskwait

            Matrix x(b_loc.rows(), 1);
            x.set_block(0, 0, x1);
            x.set_block(len_l, 0, x2);
            return x;
        }

        int mid = (node->idx_start + node->idx_end) / 2;
        int len_l = mid - node->idx_start;
        int len_r = node->idx_end - mid;

        // 1. Solve independent systems z = D^-1 * b
        Matrix z1, z2;
        #pragma omp task shared(z1)
        z1 = solve_recursive(node->left, b_loc.block(0, 0, len_l, 1));
        #pragma omp task shared(z2)
        z2 = solve_recursive(node->right, b_loc.block(len_l, 0, len_r, 1));
        // No taskwait yet, let them run while we build bases
        
        // 2. Reconstruct dense U and V for children
        Matrix U_L_dense, U_R_dense, V_L_dense, V_R_dense;
        #pragma omp task shared(U_L_dense)
        U_L_dense = reconstruct_U(node->left);
        #pragma omp task shared(U_R_dense)
        U_R_dense = reconstruct_U(node->right);
        #pragma omp task shared(V_L_dense)
        V_L_dense = reconstruct_V(node->left);
        #pragma omp task shared(V_R_dense)
        V_R_dense = reconstruct_V(node->right);
        #pragma omp taskwait // Need bases for next step

        // 3. Compute Q = D^-1 * U (Recursive solve on columns of U)
        Matrix Q1, Q2;
        #pragma omp task shared(Q1)
        Q1 = solve_multi_rhs_internal(node->left, U_L_dense);
        #pragma omp task shared(Q2)
        Q2 = solve_multi_rhs_internal(node->right, U_R_dense);
        #pragma omp taskwait // Wait for z1, z2, Q1, Q2

        // --- STABLE CORRECTION STEP ---
        // Formula: x = z - Q * B * (I + K*B)^-1 * V^T*z
        // Where K = V^T * Q
        
        // K = V^T * D^-1 * U
        Matrix VTQ1 = V_L_dense.transpose() * Q1;
        Matrix VTQ2 = V_R_dense.transpose() * Q2;
        
        Matrix K = Matrix::Zero(node->rank_V, node->rank_U);
        K.set_block(0, 0, VTQ1);
        K.set_block(node->left->rank_V, node->left->rank_U, VTQ2);

        // Term: V^T * z
        Matrix VTz_1 = V_L_dense.transpose() * z1;
        Matrix VTz_2 = V_R_dense.transpose() * z2;
        Matrix VTz = Matrix::Zero(node->rank_V, 1);
        VTz.set_block(0, 0, VTz_1);
        VTz.set_block(node->left->rank_V, 0, VTz_2);

        // Core Matrix M = I + K * B
        // This is always square (rank_V x rank_V) and well-conditioned for decay matrices
        Matrix M = K * node->B;
        for(int i=0; i<M.rows(); ++i) M(i,i) += 1.0; 

        // Solve M * gamma = VTz
        M.solve_lu(VTz); // VTz becomes gamma

        // Correction = Q * (B * gamma)
        Matrix B_gamma = node->B * VTz;
        Matrix gam1 = B_gamma.block(0, 0, node->left->rank_U, 1);
        Matrix gam2 = B_gamma.block(node->left->rank_U, 0, node->right->rank_U, 1);
        
        Matrix corr1 = Q1 * gam1;
        Matrix corr2 = Q2 * gam2;

        Matrix x(b_loc.rows(), 1);
        for (int i = 0; i < len_l; ++i) x(i, 0) = z1(i, 0) - corr1(i, 0);
        for (int i = 0; i < len_r; ++i) x(len_l + i, 0) = z2(i, 0) - corr2(i, 0);

        return x;
    }

    Matrix solve_multi_rhs_internal(HSSNode* node, const Matrix& B_mat) {
        int cols = B_mat.cols();
        Matrix Res(B_mat.rows(), cols);
        // Simple Loop is fine inside tasks; heavy work is in recursion
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