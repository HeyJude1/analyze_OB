#pragma once
#include "../blas_common.hpp"

/* Index of Minimum Absolute Value: result = argmin(|x[i]|)
 * Find the index of the element with minimum absolute value
 * 
 * This is based on OpenBLAS idamin_k implementation
 * Note: BLAS returns 1-based index, but we'll use 0-based for consistency
 */
BLASLONG correctIamin(BLASLONG n, const std::vector<double>& x, BLASLONG inc_x) {
    
    if (n <= 0 || inc_x <= 0) return 0;
    
    double minf = std::abs(x[0]);
    BLASLONG min_idx = 0;
    BLASLONG ix = inc_x;
    
    // Direct translation from OpenBLAS idamin_k (adjusted to 0-based)
    for (BLASLONG i = 1; i < n; i++) {
        if (std::abs(x[ix]) < minf) {
            min_idx = i;
            minf = std::abs(x[ix]);
        }
        ix += inc_x;
    }
    
    return min_idx;  // 0-based index
}

/* Simplified version for unit stride */
BLASLONG correctIaminSimple(BLASLONG n, const std::vector<double>& x) {
    if (n <= 0) return 0;
    
    BLASLONG min_idx = 0;
    double min_val = std::abs(x[0]);
    
    for (BLASLONG i = 1; i < n; i++) {
        if (std::abs(x[i]) < min_val) {
            min_idx = i;
            min_val = std::abs(x[i]);
        }
    }
    
    return min_idx;
}
