#pragma once
#include "../blas_common.hpp"

/* Vector Absolute Sum: result = sum(|x[i]|)
 * Compute the sum of absolute values of vector x
 * 
 * This is based on OpenBLAS dasum_k implementation
 */
double correctAsum(BLASLONG n, const std::vector<double>& x, BLASLONG inc_x) {
    
    if (n <= 0 || inc_x <= 0) return 0.0;
    
    double sumf = 0.0;
    BLASLONG ix = 0;
    
    // Direct translation from OpenBLAS dasum_k
    for (BLASLONG i = 0; i < n; i++) {
        sumf += std::abs(x[ix]);
        ix += inc_x;
    }
    
    return sumf;
}

/* Simplified version for unit stride */
double correctAsumSimple(BLASLONG n, const std::vector<double>& x) {
    double sum = 0.0;
    for (BLASLONG i = 0; i < n; i++) {
        sum += std::abs(x[i]);
    }
    return sum;
}
