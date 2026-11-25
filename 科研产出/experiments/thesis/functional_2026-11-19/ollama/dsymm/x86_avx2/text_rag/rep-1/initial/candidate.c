#include "opbench.h"

static double symmetric_at(const double *a, size_t lda, size_t i, size_t j, char uplo) {
    const int stored = uplo == 'U' ? i <= j : i >= j;
    return stored ? a[i * lda + j] : a[j * lda + i];
}

void op_dsymm(char side, char uplo, size_t m, size_t n, double alpha,
              const double *a, size_t lda, const double *b, size_t ldb,
              double beta, double *c, size_t ldc) {
    if (m == 0 || n == 0 || (alpha == 0.0 && beta == 1.0)) return;

    // Precompute pointers
    const double *a_base = a;
    const double *b_base = b;
    double *c_base = c;

    // Register blocking parameters
    const size_t reg_block_m = 8;
    const size_t reg_block_n = 8;

    for (size_t i0 = 0; i0 < m; i0 += reg_block_m) {
        for (size_t j0 = 0; j0 < n; j0 += reg_block_n) {
            double sum[reg_block_n] = {0.0};

            for (size_t p = 0; p < m; ++p) {
                const double a_val = symmetric_at(a_base, lda, i0, p, uplo);
                for (size_t j = 0; j < reg_block_n && j + j0 < n; ++j) {
                    sum[j] += a_val * b_base[p * ldb + j + j0];
                }
            }

            if (alpha != 0.0 && side == 'L') {
                for (size_t i = 0; i < reg_block_m && i + i0 < m; ++i) {
                    const size_t index = (i + i0) * ldc + j0;
                    c[index] = alpha * sum[i % reg_block_n] + (beta == 0.0 ? 0.0 : beta * c_base[index]);
                }
            } else if (alpha != 0.0) {
                for (size_t j = 0; j < reg_block_n && j + j0 < n; ++j) {
                    const size_t index = i0 * ldc + (j + j0);
                    c[index] = alpha * sum[j % reg_block_n] + (beta == 0.0 ? 0.0 : beta * c_base[index]);
                }
            }
        }
    }

    // Tail handling
    for (size_t i = m - (m % reg_block_m); i < m; ++i) {
        for (size_t j = n - (n % reg_block_n); j < n; ++j) {
            double sum = 0.0;
            if (alpha != 0.0 && side == 'L') {
                for (size_t p = 0; p < m; ++p)
                    sum += symmetric_at(a_base, lda, i, p, uplo) * b_base[p * ldb + j];
            } else if (alpha != 0.0) {
                for (size_t p = 0; p < n; ++p)
                    sum += b_base[i * ldb + p] * symmetric_at(a_base, lda, p, j, uplo);
            }
            const size_t index = i * ldc + j;
            c[index] = alpha * sum + (beta == 0.0 ? 0.0 : beta * c_base[index]);
        }
    }
}
