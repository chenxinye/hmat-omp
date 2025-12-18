#ifndef HSS_ROUTINES_OMP_H
#define HSS_ROUTINES_OMP_H

#include "utils.h"
#include "compression.h"
#include <vector>
#include <queue>
#include <algorithm>
#include <omp.h> // OpenMP header

class HSSMatrixOMP {
public:
    HSSNode* root;
    std::vector<HSSNode*> nodes;
    
    // Level-by-Level storage
    // levels[0] contains Root, levels[max_depth] contains Leaves
    std::vector<std::vector<HSSNode*>> levels; 
    int max_depth;

    int N;
    double tolerance;

    HSSMatrixOMP(double tol = 1e-6) : root(nullptr), N(0), tolerance(tol), max_depth(0) {}
    
    ~HSSMatrixOMP() { 
        for(auto n : nodes) delete n; 
    }

    // *****************************************
    // 1. Parallel Construction
    // *****************************************
    void build_from_dense(const Matrix& A, int leaf_size) {
        N = A.rows();
        root = new HSSNode();
        nodes.push_back(root);
        
        // 1. Build Topology (Serial is fine here, it's fast)
        build_topology(root, 0, N, leaf_size, 0);
        
        // 2. Organize nodes into levels for parallel traversal
        organize_levels();
        
        // 3. Parallel Compression (Bottom-Up: Max Depth -> 0)
        for (int d = max_depth; d >= 0; --d) {
            
            // Parallelize across nodes in the same level
            // schedule(dynamic) is better because workload might vary per node
            #pragma omp parallel for schedule(dynamic)
            for (size_t i = 0; i < levels[d].size(); ++i) {
                HSSNode* node = levels[d][i];
                int start = node->idx_start;
                int len = node->idx_end - node->idx_start;

                if (node->is_leaf) {
                    // Extract Diagonal
                    node->D = A.block(start, start, len, len);

                    // Extract Off-Diagonal & Compress
                    // Note: Matrix::block copies data, so it's thread-safe
                    Matrix RowBlock = A.block(start, 0, len, N);
                    Matrix ColBlockT = A.block(0, start, N, len).transpose();
                    
                    // Mask diagonal
                    for(int c=0; c<len; ++c) {
                        for(int r=0; r<len; ++r) {
                            RowBlock(r, start+c) = 0.0;
                            ColBlockT(r, start+c) = 0.0;
                        }
                    }

                    auto lr_U = Compression::compress_svd(RowBlock, tolerance);
                    auto lr_V = Compression::compress_svd(ColBlockT, tolerance);

                    node->U = lr_U.U;
                    node->V = lr_V.U; 
                } else {
                    // Merge Children
                    // Safe because children (at depth d+1) are fully processed due to implicit barrier
                    HSSNode* lc = node->left;
                    HSSNode* rc = node->right;
                    
                    int rl = lc->U.cols();
                    int rr = rc->U.cols();
                    int r_total = rl + rr;

                    // Simplified Merge
                    node->B = Matrix::Identity(r_total, r_total); 
                    node->U = Matrix::Identity(r_total, r_total);
                    node->V = Matrix::Identity(r_total, r_total);
                }
            } // End Parallel Region for this level
            // Implicit Barrier here: all threads wait before moving to depth d-1
        }
    }

    // *****************************************
    // 2. Parallel Matrix-Vector Multiplication
    // *****************************************
    Matrix multiply(const Matrix& x_vec) {
        // Shared buffer for Upward Pass
        // g must be resized to accommodate all nodes.
        // We use node->id to index. Since vector access is thread-safe for different indices.
        std::vector<Matrix> g(nodes.size()); 
        Matrix y(N, 1);

        // --- Phase 1: Upward Pass (Bottom-Up) ---
        for (int d = max_depth; d >= 0; --d) {
            #pragma omp parallel for schedule(dynamic)
            for (size_t i = 0; i < levels[d].size(); ++i) {
                HSSNode* node = levels[d][i];

                if (node->is_leaf) {
                    Matrix x_loc = x_vec.block(node->idx_start, 0, node->idx_end - node->idx_start, 1);
                    g[node->id] = node->V.transpose() * x_loc;
                } else {
                    // Children are at depth d+1, so they are ready
                    Matrix g_l = g[node->left->id];
                    Matrix g_r = g[node->right->id];
                    
                    Matrix g_stack(g_l.rows() + g_r.rows(), 1);
                    g_stack.set_block(0, 0, g_l);
                    g_stack.set_block(g_l.rows(), 0, g_r);
                    
                    // Apply B (Translation)
                    g[node->id] = g_stack; 
                }
            }
        }

        // --- Phase 2: Downward Pass (Top-Down) ---
        // (Omitted for brevity in this demo, but would mirror the above with d = 0 to max_depth)
        
        // --- Phase 3: Local Calculation (Parallel) ---
        // We can just iterate all leaves. Leaves are usually at max_depth or mixed.
        // Let's use the levels to find leaves safely.
        
        for (int d = 0; d <= max_depth; ++d) {
             #pragma omp parallel for schedule(dynamic)
             for (size_t i = 0; i < levels[d].size(); ++i) {
                HSSNode* node = levels[d][i];
                if (node->is_leaf) {
                    int len = node->idx_end - node->idx_start;
                    Matrix x_loc = x_vec.block(node->idx_start, 0, len, 1);
                    Matrix res = node->D * x_loc;
                    
                    // Writing to 'y' requires care if we use block operations?
                    // No, distinct leaves write to distinct rows of y. Thread-safe.
                    y.set_block(node->idx_start, 0, res);
                }
             }
        }

        return y;
    }

    // *****************************************
    // 3. Parallel Solver (Simplified)
    // *****************************************
    Matrix solve(const Matrix& b) {
        Matrix x = b; 
        
        // Parallel Block-Jacobi (Leaf Solve)
        for (int d = 0; d <= max_depth; ++d) {
            #pragma omp parallel for schedule(dynamic)
            for (size_t i = 0; i < levels[d].size(); ++i) {
                HSSNode* node = levels[d][i];
                if (node->is_leaf) {
                    int start = node->idx_start;
                    int len = node->idx_end - node->idx_start;
                    
                    // Since x is shared, we must be careful.
                    // Reading from 'x' is fine.
                    // Writing to 'x' at distinct 'start' indices is thread-safe.
                    Matrix b_loc = x.block(start, 0, len, 1);
                    
                    // Local Solve
                    Matrix D_copy = node->D;
                    D_copy.solve_lu(b_loc); 
                    
                    x.set_block(start, 0, b_loc);
                }
            }
        }
        return x;
    }

private:
    void build_topology(HSSNode* node, int start, int end, int leaf_size, int depth) {
        node->idx_start = start;
        node->idx_end = end;
        node->id = nodes.size() - 1; // ID based on creation order
        
        // Track max depth
        if (depth > max_depth) max_depth = depth;

        if ((end - start) <= leaf_size) {
            node->is_leaf = true;
        } else {
            int mid = (start + end) / 2;
            node->left = new HSSNode(); nodes.push_back(node->left);
            build_topology(node->left, start, mid, leaf_size, depth + 1);
            
            node->right = new HSSNode(); nodes.push_back(node->right);
            build_topology(node->right, mid, end, leaf_size, depth + 1);
            
            node->left->parent = node;
            node->right->parent = node;
        }
    }

    // BFS to organize nodes into levels
    void organize_levels() {
        levels.resize(max_depth + 1);
        
        std::queue<std::pair<HSSNode*, int>> q;
        if(root) q.push({root, 0});
        
        while(!q.empty()) {
            auto current = q.front();
            q.pop();
            
            HSSNode* node = current.first;
            int depth = current.second;
            
            levels[depth].push_back(node);
            
            if(node->left) q.push({node->left, depth+1});
            if(node->right) q.push({node->right, depth+1});
        }
    }
};

#endif