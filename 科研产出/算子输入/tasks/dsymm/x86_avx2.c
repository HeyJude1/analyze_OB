#include "opbench.h"
#include <immintrin.h>

static double symmetric_at(const double *a, size_t lda, size_t i, size_t j, char uplo) {
    const int stored = uplo == 'U' ? i <= j : i >= j;
    return stored ? a[i * lda + j] : a[j * lda + i];
}

void op_dsymm(char side, char uplo, size_t m, size_t n, double alpha,
              const double *a, size_t lda, const double *b, size_t ldb,
              double beta, double *c, size_t ldc) {
    if (m == 0 || n == 0 || (alpha == 0.0 && beta == 1.0)) return;
    const __m256d av = _mm256_set1_pd(alpha), bv = _mm256_set1_pd(beta);
    for (size_t i = 0; i < m; ++i) {
        size_t j = 0;
        for (; n - j >= 4; j += 4) {
            __m256d acc = _mm256_setzero_pd();
            if (alpha != 0.0 && side == 'L') {
                for (size_t p = 0; p < m; ++p) {
                    const __m256d factor = _mm256_set1_pd(symmetric_at(a, lda, i, p, uplo));
                    acc = _mm256_fmadd_pd(factor, _mm256_loadu_pd(b + p * ldb + j), acc);
                }
            } else if (alpha != 0.0) {
                for (size_t p = 0; p < n; ++p) {
                    const __m256d row = _mm256_setr_pd(symmetric_at(a, lda, p, j, uplo),
                        symmetric_at(a, lda, p, j + 1, uplo), symmetric_at(a, lda, p, j + 2, uplo),
                        symmetric_at(a, lda, p, j + 3, uplo));
                    acc = _mm256_fmadd_pd(_mm256_set1_pd(b[i * ldb + p]), row, acc);
                }
            }
            double *out = c + i * ldc + j;
            acc = _mm256_mul_pd(av, acc);
            if (beta != 0.0) acc = _mm256_fmadd_pd(bv, _mm256_loadu_pd(out), acc);
            _mm256_storeu_pd(out, acc);
        }
        for (; j < n; ++j) {
            double sum = 0.0;
            if (alpha != 0.0 && side == 'L') {
                for (size_t p = 0; p < m; ++p)
                    sum += symmetric_at(a, lda, i, p, uplo) * b[p * ldb + j];
            } else if (alpha != 0.0) {
                for (size_t p = 0; p < n; ++p)
                    sum += b[i * ldb + p] * symmetric_at(a, lda, p, j, uplo);
            }
            double *out = c + i * ldc + j;
            *out = alpha * sum + (beta == 0.0 ? 0.0 : beta * *out);
        }
    }
}
