#pragma once
#include "../blas_common.hpp"

/* Vector Dot Product: result = x^T * y = sum(x[i] * y[i])
 * x and y are n-element vectors
 * 
 * This is based on OpenBLAS ddot_k implementation
 */
double correctDot(BLASLONG n, const std::vector<double>& x, BLASLONG inc_x,
                  const std::vector<double>& y, BLASLONG inc_y) {
    
    if (n < 0) return 0.0;
    
    double dot = 0.0;
    BLASLONG ix = 0, iy = 0;
    
    // Direct translation from OpenBLAS ddot_k
    for (BLASLONG i = 0; i < n; i++) {
        dot += x[ix] * y[iy];
        ix += inc_x;
        iy += inc_y;
    }
    
    return dot;
}

/* Simplified version for unit stride */
double correctDotSimple(BLASLONG n, const std::vector<double>& x, const std::vector<double>& y) {
    double dot = 0.0;
    
    for (BLASLONG i = 0; i < n; i++) {
        dot += x[i] * y[i];
    }
    
    return dot;
}

/* Numerically stable version using Kahan summation */
double correctDotStable(BLASLONG n, const std::vector<double>& x, const std::vector<double>& y) {
    double sum = 0.0;
    double c = 0.0;  // Compensation for lost low-order bits
    
    for (BLASLONG i = 0; i < n; i++) {
        double product = x[i] * y[i];
        double t = sum + product;
        c += (sum - t) + product;  // Algebraically, c should always be zero
        sum = t;
    }
    
    return sum + c;
}
