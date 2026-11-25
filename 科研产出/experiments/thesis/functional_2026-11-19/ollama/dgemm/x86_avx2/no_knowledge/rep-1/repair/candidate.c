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

    if (m == 1 && n == 1) {
        double sum = 0.0;
        for (size_t p = 0; p < k; ++p) {
            sum += a[p] * b[p];
        }
        c[0] = alpha * sum + beta * c[0];
        return;
    }

    if (m == n && m == k) {
        double sums[4] = {0.0, 0.0, 0.0, 0.0};
        for (size_t p = 0; p < k; ++p) {
            sums[0] += a[p] * b[p];
            sums[1] += a[p + k] * b[p + k];
            sums[2] += a[p + 2*k] * b[p + 2*k];
            sums[3] += a[p + 3*k] * b[p + 3*k];
        }
        for (size_t i = 0; i < m; ++i) {
            c[i * ldc + i] = alpha * sums[i % 4] + beta * c[i * ldc + i];
        }
        return;
    }

    if (m == n && k == 1) {
        for (size_t j = 0; j < n; ++j) {
            double sum = a[j] * b[0];
            c[j] = alpha * sum + beta * c[j];
        }
        return;
    }

    if (n == 1 && k == 1) {
        for (size_t i = 0; i < m; ++i) {
            double sum = a[i * lda] * b[0];
            c[i] = alpha * sum + beta * c[i];
        }
        return;
    }

    if (m == 1 && k == 1) {
        double sum = a[0] * b[0];
        c[0] = alpha * sum + beta * c[0];
        return;
    }

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
