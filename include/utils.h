#ifndef UTILS_H
#define UTILS_H

#include "matrix.h"

//  HSS Tree Node Structure 
struct HSSNode {
    int id;
    int idx_start, idx_end; // Global indices range
    bool is_leaf;
    
    // Tree Pointers
    HSSNode *left = nullptr;
    HSSNode *right = nullptr;
    HSSNode *parent = nullptr;

    //  HSS Generators 
    Matrix D; // Diagonal block (Leaf only)
    Matrix U; // Row basis
    Matrix V; // Column basis 
    Matrix B; // Translation operator

    //  Factorization Data (ULV) 
    Matrix Q; // Orthogonal factors
    Matrix L; // Reduced/Triangular factors
    Matrix W; // Merge buffer (Schur complement)

    HSSNode() : is_leaf(false) {}
};

#endif // UTILS_H