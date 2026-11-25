#include "opbench.h"

void op_dgemm(size_t m, size_t n, size_t k, double alpha, const double *a, size_t lda,
              const double *b, size_t ldb, double beta, double *c, size_t ldc) {
    if (alpha == 0.0 || k == 0) {
        if (beta != 0.0) {
            for (size_t i = 0; i < m; ++i) {
                for (size_t j = 0; j < n; ++j) {
                    c[i * ldc + j] *= beta;
                }
            }
        }
        return;
    }

    if (beta == 0.0) {
        for (size_t i = 0; i < m; ++i) {
            for (size_t j = 0; j < n; ++j) {
                double sum = 0.0;
                for (size_t p = 0; p < k; ++p) {
                    sum += a[i * lda + p] * b[p * ldb + j];
                }
                c[i * ldc + j] = alpha * sum;
            }
        }
    } else {
        for (size_t i = 0; i < m; ++i) {
            for (size_t j = 0; j < n; ++j) {
                double sum = 0.0;
                for (size_t p = 0; p < k; ++p) {
                    sum += a[i * lda + p] * b[p * ldb + j];
                }
                c[i * ldc + j] = alpha * sum + beta * c[i * ldc + j];
            }
        }
    }
}
