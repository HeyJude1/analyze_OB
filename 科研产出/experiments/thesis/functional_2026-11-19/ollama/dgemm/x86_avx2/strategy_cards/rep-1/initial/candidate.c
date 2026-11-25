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

    const size_t block_size = 4;
    const double init_value = 0.0;

    for (size_t i = 0; i < m; i += block_size) {
        for (size_t j = 0; j < n; j += block_size) {
            double acc[block_size][block_size] = {init_value};

            for (size_t p = 0; p < k; ++p) {
                const size_t ai = i * lda + p;
                const size_t bi = p * ldb + j;

                for (size_t ii = 0; ii < block_size && i + ii < m; ++ii) {
                    for (size_t jj = 0; jj < block_size && j + jj < n; ++jj) {
                        acc[ii][jj] += a[ai + ii * lda] * b[bi + jj];
                    }
                }
            }

            for (size_t ii = 0; ii < block_size && i + ii < m; ++ii) {
                for (size_t jj = 0; jj < block_size && j + jj < n; ++jj) {
                    const size_t out = (i + ii) * ldc + (j + jj);
                    c[out] = alpha * acc[ii][jj] + (beta == 0.0 ? 0.0 : beta * c[out]);
                }
            }
        }
    }

    // Handle tail
    for (size_t i = m - (m % block_size); i < m; ++i) {
        for (size_t j = n - (n % block_size); j < n; ++j) {
            double sum = 0.0;
            for (size_t p = 0; p < k; ++p) {
                sum += a[i * lda + p] * b[p * ldb + j];
            }
            const size_t out = i * ldc + j;
            c[out] = alpha * sum + (beta == 0.0 ? 0.0 : beta * c[out]);
        }
    }
}
