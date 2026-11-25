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

    const double *pa = a;
    double *pc = c;

    if (m % 4 == 0 && n % 4 == 0) {
        for (size_t j = 0; j < n; j += 4) {
            for (size_t i = 0; i < m; i++) {
                double sum1 = pa[i * lda] * b[j * ldb];
                double sum2 = pa[i * lda + 1] * b[(j + 1) * ldb];
                double sum3 = pa[i * lda + 2] * b[(j + 2) * ldb];
                double sum4 = pa[i * lda + 3] * b[(j + 3) * ldb];

                pc[i * ldc + j] += alpha * sum1;
                pc[i * ldc + j + 1] += alpha * sum2;
                pc[i * ldc + j + 2] += alpha * sum3;
                pc[i * ldc + j + 3] += alpha * sum4;
            }
            pa += 4 * lda;
        }
    } else {
        for (size_t j = 0; j < n; ++j) {
            for (size_t i = 0; i < m; i++) {
                double sum = 0.0;
                for (size_t p = 0; p < k; ++p) sum += pa[i * lda + p] * b[p * ldb + j];
                pc[i * ldc + j] += alpha * sum;
            }
        }
    }

    if (beta != 0.0) {
        for (size_t i = 0; i < m; ++i) {
            for (size_t j = 0; j < n; ++j) {
                pc[i * ldc + j] *= beta;
            }
        }
    }
}
