#pragma once
#include "../blas_common.hpp"

/* Vector Swap: swap(x, y)
 * Swap the contents of vectors x and y
 * 
 * This is based on OpenBLAS dswap_k implementation
 */
void correctSwap(BLASLONG n, std::vector<double>& x, BLASLONG inc_x,
                 std::vector<double>& y, BLASLONG inc_y) {
    
    if (n < 0) return;
    
    BLASLONG ix = 0, iy = 0;
    
    // Direct translation from OpenBLAS dswap_k
    for (BLASLONG i = 0; i < n; i++) {
        double temp = x[ix];
        x[ix] = y[iy];
        y[iy] = temp;
        ix += inc_x;
        iy += inc_y;
    }
}

/* Simplified version for unit stride */
void correctSwapSimple(BLASLONG n, std::vector<double>& x, std::vector<double>& y) {
    for (BLASLONG i = 0; i < n; i++) {
        std::swap(x[i], y[i]);
    }
}
