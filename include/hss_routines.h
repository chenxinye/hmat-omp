#ifndef HSS_ROUTINES_H
#define HSS_ROUTINES_H

#include "utils.h"
#include "compression.h"
#include <stack>
#include <vector>

class HSSMatrix {
public:
    HSSNode* root;
    std::vector<HSSNode*> nodes;
    
    // Stack-based traversal orders (Non-recursive)
    std::vector<HSSNode*> stack_post_order; // Bottom-Up (Leaves -> Root)
    std::vector<HSSNode*> stack_pre_order;  // Top-Down (Root -> Leaves)
    
    int N;
    double tolerance;

    HSSMatrix(double tol = 1e-6) : root(nullptr), N(0), tolerance(tol) {}
    
    ~HSSMatrix() { 
        for(auto n : nodes) delete n; 
    }

    // *****************************************
    // 1. Construction Logic
    // *****************************************
    void build_from_dense(const Matrix& A, int leaf_size) {
        N = A.rows();
        root = new HSSNode();
        nodes.push_back(root);
        
        // 1.1 Topology
        build_topology(root, 0, N, leaf_size);
        generate_orders();
        
        // 1.2 Compression (Bottom-Up)
        for (HSSNode* node : stack_post_order) {
            int start = node->idx_start;
            int len = node->idx_end - node->idx_start;

            if (node->is_leaf) {
                // Diagonal block
                node->D = A.block(start, start, len, len);

                // Off-diagonal extraction
                // Mask diagonal block to 0 for compression
                Matrix RowBlock = A.block(start, 0, len, N);
                Matrix ColBlockT = A.block(0, start, N, len).transpose();
                
                for(int j=0; j<len; ++j) {
                    for(int i=0; i<len; ++i) {
                        RowBlock(i, start+j) = 0.0;
                        ColBlockT(i, start+j) = 0.0;
                    }
                }

                auto lr_U = Compression::compress_svd(RowBlock, tolerance);
                auto lr_V = Compression::compress_svd(ColBlockT, tolerance);

                node->U = lr_U.U;
                node->V = lr_V.U; 
            } else {
                // Merge Children
                HSSNode* lc = node->left;
                HSSNode* rc = node->right;
                
                int rl = lc->U.cols();
                int rr = rc->U.cols();
                int r_total = rl + rr;

                // Simplified merge (Identity Transfer)
                // Real implementation would recompress [U_left, 0; 0, U_right]
                node->B = Matrix::Identity(r_total, r_total); 
                node->U = Matrix::Identity(r_total, r_total);
                node->V = Matrix::Identity(r_total, r_total);
            }
        }
    }

    // *****************************************
    // 2. Matrix-Vector Multiplication (Fast GEMV)
    // *****************************************
    Matrix multiply(const Matrix& x_vec) {
        std::vector<Matrix> g(nodes.size()); 
        Matrix y(N, 1);

        // Upward Pass
        for (HSSNode* node : stack_post_order) {
            if (node->is_leaf) {
                Matrix x_loc = x_vec.block(node->idx_start, 0, node->idx_end - node->idx_start, 1);
                g[node->id] = node->V.transpose() * x_loc;
            } else {
                Matrix g_l = g[node->left->id];
                Matrix g_r = g[node->right->id];
                // Stack children vectors
                Matrix g_stack(g_l.rows() + g_r.rows(), 1);
                g_stack.set_block(0, 0, g_l);
                g_stack.set_block(g_l.rows(), 0, g_r);
                g[node->id] = g_stack; // Apply B here in full version
            }
        }

        // Downward Pass (omitted for brevity, typically computes 'f')
        // ...

        // Local Calculation (Diagonal)
        for (HSSNode* node : stack_post_order) {
            if (node->is_leaf) {
                int len = node->idx_end - node->idx_start;
                Matrix x_loc = x_vec.block(node->idx_start, 0, len, 1);
                Matrix res = node->D * x_loc;
                y.set_block(node->idx_start, 0, res);
            }
        }
        return y;
    }

    // *****************************************
    // 3. Solver (ULV / LU Wrapper)
    // *****************************************
    // A simplified solver that treats diagonal blocks with dense LU
    Matrix solve(const Matrix& b) {
        Matrix x = b; 
        
        // Very simplified Block-Jacobi preconditioner style for demo
        // (Real HSS solver requires full ULV forward/backward)
        for (HSSNode* node : stack_post_order) {
            if (node->is_leaf) {
                int start = node->idx_start;
                int len = node->idx_end - node->idx_start;
                Matrix b_loc = x.block(start, 0, len, 1);
                
                // Solve locally using LU
                Matrix D_copy = node->D;
                D_copy.solve_lu(b_loc); 
                
                x.set_block(start, 0, b_loc);
            }
        }
        return x;
    }

private:
    void build_topology(HSSNode* node, int start, int end, int leaf_size) {
        node->idx_start = start;
        node->idx_end = end;
        node->id = nodes.size() - 1;
        if ((end - start) <= leaf_size) {
            node->is_leaf = true;
        } else {
            int mid = (start + end) / 2;
            node->left = new HSSNode(); nodes.push_back(node->left);
            build_topology(node->left, start, mid, leaf_size);
            node->right = new HSSNode(); nodes.push_back(node->right);
            build_topology(node->right, mid, end, leaf_size);
        }
    }

    void generate_orders() {
        stack_post_order.clear();
        std::stack<HSSNode*> s;
        std::stack<HSSNode*> out;
        if(root) s.push(root);
        while (!s.empty()) {
            HSSNode* curr = s.top(); s.pop();
            out.push(curr);
            if (curr->left) s.push(curr->left);
            if (curr->right) s.push(curr->right);
        }
        while (!out.empty()) {
            stack_post_order.push_back(out.top());
            out.pop();
        }
    }
};

#endif // HSS_ROUTINES_H