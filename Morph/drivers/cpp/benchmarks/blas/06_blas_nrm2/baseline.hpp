#pragma once
#include "../blas_common.hpp"

/* Vector 2-Norm: result = ||x||_2 = sqrt(sum(x[i]^2))
 * Compute the Euclidean norm of vector x
 * 
 * This is based on OpenBLAS dnrm2_k implementation with numerical stability
 */
double correctNrm2(BLASLONG n, const std::vector<double>& x, BLASLONG inc_x) {
    
    if (n <= 0 || inc_x == 0) return 0.0;
    if (n == 1) return std::abs(x[0]);
    
    // Use scale and sum-of-squares algorithm for numerical stability (from OpenBLAS)
    double scale = 0.0;
    double ssq = 1.0;
    
    BLASLONG ix = 0;
    for (BLASLONG i = 0; i < n; i++) {
        if (x[ix] != 0.0) {
            double absxi = std::abs(x[ix]);
            if (scale < absxi) {
                double temp = scale / absxi;
                ssq = 1.0 + ssq * temp * temp;
                scale = absxi;
            } else {
                double temp = absxi / scale;
                ssq += temp * temp;
            }
        }
        ix += inc_x;
    }
    
    return scale * std::sqrt(ssq);
}

/* Simple version (may overflow/underflow) */
double correctNrm2Simple(BLASLONG n, const std::vector<double>& x) {
    double sum = 0.0;
    for (BLASLONG i = 0; i < n; i++) {
        sum += x[i] * x[i];
    }
    return std::sqrt(sum);
}
