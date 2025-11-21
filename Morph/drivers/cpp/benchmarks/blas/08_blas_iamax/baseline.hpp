#pragma once
#include "../blas_common.hpp"

/* Index of Maximum Absolute Value: result = argmax(|x[i]|)
 * Find the index of the element with maximum absolute value
 * 
 * This is based on OpenBLAS idamax_k implementation
 * Note: BLAS returns 1-based index, but we'll use 0-based for consistency
 */
BLASLONG correctIamax(BLASLONG n, const std::vector<double>& x, BLASLONG inc_x) {
    
    if (n <= 0 || inc_x <= 0) return 0;
    
    double maxf = std::abs(x[0]);
    BLASLONG max_idx = 0;
    BLASLONG ix = inc_x;
    
    // Direct translation from OpenBLAS idamax_k (adjusted to 0-based)
    for (BLASLONG i = 1; i < n; i++) {
        if (std::abs(x[ix]) > maxf) {
            max_idx = i;
            maxf = std::abs(x[ix]);
        }
        ix += inc_x;
    }
    
    return max_idx;  // 0-based index
}

/* Simplified version for unit stride */
BLASLONG correctIamaxSimple(BLASLONG n, const std::vector<double>& x) {
    if (n <= 0) return 0;
    
    BLASLONG max_idx = 0;
    double max_val = std::abs(x[0]);
    
    for (BLASLONG i = 1; i < n; i++) {
        if (std::abs(x[i]) > max_val) {
            max_idx = i;
            max_val = std::abs(x[i]);
        }
    }
    
    return max_idx;
}
