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
    const size_t reg_block_m = 4;
    const size_t reg_block_n = 4;

    for (size_t i0 = 0; i0 < m; i0 += reg_block_m) {
        for (size_t j0 = 0; j0 < n; j0 += reg_block_n) {
            double sum[reg_block_m * reg_block_n];

            for (size_t p = 0; p < m; ++p) {
                const double a_val = symmetric_at(a_base, lda, i0 + p, p, uplo);
                for (size_t j = j0; j < n && j < j0 + reg_block_n; ++j) {
                    sum[(i0 + p - i0) * reg_block_n + (j - j0)] += a_val * b_base[p * ldb + j];
                }
            }

            for (size_t i = i0; i < m && i < i0 + reg_block_m; ++i) {
                const size_t index = i * ldc + j0;
                c[index] = alpha * sum[(i - i0) * reg_block_n + 0] +
                           (beta == 0.0 ? 0.0 : beta * c_base[index]);
            }
        }
    }
}
