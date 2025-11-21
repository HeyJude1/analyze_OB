#pragma once
#include "../blas_common.hpp"

/* Vector Scale and Add: y = alpha * x + beta * y
 * x and y are n-element vectors
 * 
 * This is based on OpenBLAS daxpby_k implementation
 */
void correctAxpby(BLASLONG n, double alpha, const std::vector<double>& x, BLASLONG inc_x,
                  double beta, std::vector<double>& y, BLASLONG inc_y) {
    
    if (n < 0) return;
    
    BLASLONG ix = 0, iy = 0;
    
    // Handle special cases for optimization (from OpenBLAS)
    if (beta == 0.0) {
        if (alpha == 0.0) {
            // y = 0
            for (BLASLONG i = 0; i < n; i++) {
                y[iy] = 0.0;
                iy += inc_y;
            }
        } else {
            // y = alpha * x
            for (BLASLONG i = 0; i < n; i++) {
                y[iy] = alpha * x[ix];
                ix += inc_x;
                iy += inc_y;
            }
        }
    } else {
        if (alpha == 0.0) {
            // y = beta * y
            for (BLASLONG i = 0; i < n; i++) {
                y[iy] *= beta;
                iy += inc_y;
            }
        } else {
            // y = alpha * x + beta * y (general case)
            for (BLASLONG i = 0; i < n; i++) {
                y[iy] = alpha * x[ix] + beta * y[iy];
                ix += inc_x;
                iy += inc_y;
            }
        }
    }
}

/* Simplified version for unit stride */
void correctAxpbySimple(BLASLONG n, double alpha, const std::vector<double>& x,
                        double beta, std::vector<double>& y) {
    
    // Unit stride version
    if (beta == 0.0) {
        if (alpha == 0.0) {
            std::fill(y.begin(), y.begin() + n, 0.0);
        } else {
            for (BLASLONG i = 0; i < n; i++) {
                y[i] = alpha * x[i];
            }
        }
    } else {
        if (alpha == 0.0) {
            for (BLASLONG i = 0; i < n; i++) {
                y[i] *= beta;
            }
        } else {
            for (BLASLONG i = 0; i < n; i++) {
                y[i] = alpha * x[i] + beta * y[i];
            }
        }
    }
}
