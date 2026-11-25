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

    size_t bm = m, bn = n, bk = k;
    if (bm > 16) bm = 16;
    if (bn > 16) bn = 16;
    if (bk > 16) bk = 16;

    for (size_t j = 0; j < bn; j += bn) {
        for (size_t i = 0; i < bm; i += bm) {
            double acc[bm][bn];
            for (size_t ii = 0; ii < bm && i + ii < m; ++ii) {
                for (size_t jj = 0; jj < bn && j + jj < n; ++jj) {
                    acc[ii][jj] = 0.0;
                }
            }
            for (size_t k = 0; k < bk; k++) {
                for (size_t ii = 0; ii < bm && i + ii < m; ++ii) {
                    for (size_t jj = 0; jj < bn && j + jj < n; ++jj) {
                        acc[ii][jj] += pa[(i + ii) * lda + k] * b[k * ldb + j + jj];
                    }
                }
            }
            for (size_t ii = 0; ii < bm && i + ii < m; ++ii) {
                for (size_t jj = 0; jj < bn && j + jj < n; ++jj) {
                    pc[(i + ii) * ldc + j + jj] += alpha * acc[ii][jj];
                }
            }
        }
        pa += bm * lda;
        pc += bm * ldc;
    }

    if (m % bm != 0 || n % bn != 0) {
        for (size_t i = m - (m % bm); i < m; ++i) {
            for (size_t j = n - (n % bn); j < n; ++j) {
                double sum = 0.0;
                for (size_t p = 0; p < k; ++p) {
                    sum += a[i * lda + p] * b[p * ldb + j];
                }
                pc[i * ldc + j] = alpha * sum + beta * pc[i * ldc + j];
            }
        }
    }
}
