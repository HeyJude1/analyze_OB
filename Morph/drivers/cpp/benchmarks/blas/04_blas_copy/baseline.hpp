#pragma once
#include "../blas_common.hpp"

/* Vector Copy: y = x
 * Copy n elements from vector x to vector y
 * 
 * This is based on OpenBLAS dcopy_k implementation
 */
void correctCopy(BLASLONG n, const std::vector<double>& x, BLASLONG inc_x,
                 std::vector<double>& y, BLASLONG inc_y) {
    
    if (n < 0) return;
    
    BLASLONG ix = 0, iy = 0;
    
    // Direct translation from OpenBLAS dcopy_k
    for (BLASLONG i = 0; i < n; i++) {
        y[iy] = x[ix];
        ix += inc_x;
        iy += inc_y;
    }
}

/* Simplified version for unit stride */
void correctCopySimple(BLASLONG n, const std::vector<double>& x, std::vector<double>& y) {
    for (BLASLONG i = 0; i < n; i++) {
        y[i] = x[i];
    }
}

/* STL version for comparison */
void correctCopySTL(BLASLONG n, const std::vector<double>& x, std::vector<double>& y) {
    std::copy(x.begin(), x.begin() + n, y.begin());
}
