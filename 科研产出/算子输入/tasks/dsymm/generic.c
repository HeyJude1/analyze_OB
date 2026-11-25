#include "opbench.h"

static double symmetric_at(const double *a, size_t lda, size_t i, size_t j, char uplo) {
    const int stored = uplo == 'U' ? i <= j : i >= j;
    return stored ? a[i * lda + j] : a[j * lda + i];
}

void op_dsymm(char side, char uplo, size_t m, size_t n, double alpha,
              const double *a, size_t lda, const double *b, size_t ldb,
              double beta, double *c, size_t ldc) {
    if (m == 0 || n == 0 || (alpha == 0.0 && beta == 1.0)) return;
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            if (alpha != 0.0 && side == 'L') {
                for (size_t p = 0; p < m; ++p)
                    sum += symmetric_at(a, lda, i, p, uplo) * b[p * ldb + j];
            } else if (alpha != 0.0) {
                for (size_t p = 0; p < n; ++p)
                    sum += b[i * ldb + p] * symmetric_at(a, lda, p, j, uplo);
            }
            const size_t index = i * ldc + j;
            c[index] = alpha * sum + (beta == 0.0 ? 0.0 : beta * c[index]);
        }
    }
}
