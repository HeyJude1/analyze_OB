#pragma once
#include "../blas_common.hpp"

/* Matrix-Matrix Multiplication: C = alpha * A * B + beta * C
 * A is M x K matrix (row-major)
 * B is K x N matrix (row-major) 
 * C is M x N matrix (row-major)
 * 
 * This is based on OpenBLAS dgemm_small_kernel_b0_nn implementation
 */
void correctGemm(BLASLONG M, BLASLONG N, BLASLONG K, 
                 double alpha, const std::vector<double>& A, BLASLONG lda,
                 const std::vector<double>& B, BLASLONG ldb,
                 double beta, std::vector<double>& C, BLASLONG ldc) {
    
    // Handle beta scaling first
    if (beta == 0.0) {
        std::fill(C.begin(), C.end(), 0.0);
    } else if (beta != 1.0) {
        for (BLASLONG i = 0; i < M * N; i++) {
            C[i] *= beta;
        }
    }
    
    // Compute C += alpha * A * B (simplified from OpenBLAS)
    for (BLASLONG i = 0; i < M; i++) {
        for (BLASLONG j = 0; j < N; j++) {
            double result = 0.0;
            for (BLASLONG k = 0; k < K; k++) {
                result += A[i * lda + k] * B[k * ldb + j];
            }
            C[i * ldc + j] += alpha * result;
        }
    }
}

/* Simplified version for small matrices (directly from OpenBLAS) */
void correctGemmSimple(BLASLONG M, BLASLONG N, BLASLONG K, 
                       double alpha, const std::vector<double>& A, BLASLONG lda,
                       const std::vector<double>& B, BLASLONG ldb,
                       std::vector<double>& C, BLASLONG ldc) {
    
    // Direct translation of OpenBLAS dgemm_small_kernel_b0_nn
    for (BLASLONG i = 0; i < M; i++) {
        for (BLASLONG j = 0; j < N; j++) {
            double result = 0.0;
            for (BLASLONG k = 0; k < K; k++) {
                result += A[i + k * lda] * B[k + j * ldb];
            }
            C[i + j * ldc] = alpha * result;
        }
    }
}
