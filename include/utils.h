#ifndef UTILS_H
#define UTILS_H

#include "matrix.h"

struct HSSNode {
    int id;
    int idx_start, idx_end; 
    bool is_leaf;
    HSSNode *left = nullptr;
    HSSNode *right = nullptr;

    // Generators
    Matrix D; // Diagonal Block (Leaf only)
    
    // Off-Diagonal Interaction: 
    // Top-Right Block A12 approx U_L * B_12 * V_R^T
    // Bottom-Left Block A21 approx U_R * B_21 * V_L^T
    
    // Bases stored at nodes
    Matrix U; // Row Basis
    Matrix V; // Col Basis

    // Coupling Matrices (Scaling Factors)
    Matrix B_12; 
    Matrix B_21;

    HSSNode() : is_leaf(false) {}
};

#endif