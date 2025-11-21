#pragma once
#include "../blas_common.hpp"

/* Matrix-Vector Multiplication: y = alpha * A * x + beta * y
 * A is M x N matrix (row-major)
 * x is N-element vector
 * y is M-element vector
 * 
 * This is based on OpenBLAS dgemv_n implementation
 */
void correctGemv(BLASLONG m, BLASLONG n, double alpha, 
                 const std::vector<double>& A, BLASLONG lda,
                 const std::vector<double>& x, BLASLONG inc_x,
                 double beta, std::vector<double>& y, BLASLONG inc_y) {
    
    // Handle beta scaling first
    if (beta == 0.0) {
        for (BLASLONG i = 0; i < m; i++) {
            y[i * inc_y] = 0.0;
        }
    } else if (beta != 1.0) {
        for (BLASLONG i = 0; i < m; i++) {
            y[i * inc_y] *= beta;
        }
    }
    
    // Compute y += alpha * A * x (based on OpenBLAS dgemv_n)
    BLASLONG ix = 0;
    const double* a_ptr = A.data();
    
    for (BLASLONG j = 0; j < n; j++) {
        double temp = alpha * x[ix];
        BLASLONG iy = 0;
        
        for (BLASLONG i = 0; i < m; i++) {
            y[iy] += temp * a_ptr[i];
            iy += inc_y;
        }
        
        a_ptr += lda;
        ix += inc_x;
    }
}

/* Simplified version for unit stride (directly from OpenBLAS) */
void correctGemvSimple(BLASLONG m, BLASLONG n, double alpha,
                       const std::vector<double>& A, BLASLONG lda,
                       const std::vector<double>& x, std::vector<double>& y) {
    
    // Direct translation of OpenBLAS dgemv_n with inc_x=1, inc_y=1
    for (BLASLONG j = 0; j < n; j++) {
        double temp = alpha * x[j];
        for (BLASLONG i = 0; i < m; i++) {
            y[i] += temp * A[j * lda + i];  // Column-major access
        }
    }
}

/* Row-major version for better cache performance */
void correctGemvRowMajor(BLASLONG m, BLASLONG n, double alpha,
                         const std::vector<double>& A, BLASLONG lda,
                         const std::vector<double>& x, std::vector<double>& y) {
    
    // Row-major matrix-vector multiplication
    for (BLASLONG i = 0; i < m; i++) {
        double sum = 0.0;
        for (BLASLONG j = 0; j < n; j++) {
            sum += A[i * lda + j] * x[j];
        }
        y[i] += alpha * sum;
    }
}
