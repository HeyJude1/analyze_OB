#include "opbench.h"

static double symmetric_at(const double *a, size_t lda, size_t i, size_t j, char uplo) {
    const int stored = uplo == 'U' ? i <= j : i >= j;
    return stored ? a[i * lda + j] : a[j * lda + i];
}

void op_dsymm(char side, char uplo, size_t m, size_t n, double alpha,
              const double *a, size_t lda, const double *b, size_t ldb,
              double beta, double *c, size_t ldc) {
    if (m == 0 || n == 0 || (alpha == 0.0 && beta == 1.0)) return;

    // Loop unroll factor
    const int K_UNROLL = 4;

    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; j += K_UNROLL) {
            double sum[K_UNROLL] = {0.0};
            if (alpha != 0.0 && side == 'L') {
                for (size_t p = 0; p < m; ++p)
                    for (int k = 0; k < K_UNROLL; ++k)
                        sum[k] += symmetric_at(a, lda, i, p, uplo) * b[p * ldb + j + k];
            } else if (alpha != 0.0) {
                for (size_t p = 0; p < n; ++p)
                    for (int k = 0; k < K_UNROLL; ++k)
                        sum[k] += b[i * ldb + p] * symmetric_at(a, lda, p, j + k, uplo);
            }
            for (int k = 0; k < K_UNROLL; ++k) {
                const size_t index = i * ldc + j + k;
                c[index] = alpha * sum[k] + (beta == 0.0 ? 0.0 : beta * c[index]);
            }
        }
    }
}
