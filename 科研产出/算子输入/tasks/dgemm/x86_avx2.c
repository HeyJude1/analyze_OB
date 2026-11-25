#include "opbench.h"
#include <immintrin.h>

void op_dgemm(size_t m, size_t n, size_t k, double alpha, const double *a, size_t lda,
              const double *b, size_t ldb, double beta, double *c, size_t ldc) {
    const __m256d av = _mm256_set1_pd(alpha);
    const __m256d bv = _mm256_set1_pd(beta);
    for (size_t i = 0; i < m; ++i) {
        size_t j = 0;
        for (; n - j >= 8; j += 8) {
            __m256d p0 = _mm256_setzero_pd();
            __m256d p1 = _mm256_setzero_pd();
            if (alpha != 0.0) {
                for (size_t p = 0; p < k; ++p) {
                    const __m256d factor = _mm256_set1_pd(a[i * lda + p]);
                    p0 = _mm256_fmadd_pd(factor, _mm256_loadu_pd(b + p * ldb + j), p0);
                    p1 = _mm256_fmadd_pd(factor, _mm256_loadu_pd(b + p * ldb + j + 4), p1);
                }
            }
            if (k != 0) {
                p0 = _mm256_mul_pd(av, p0);
                p1 = _mm256_mul_pd(av, p1);
            }
            double *out = c + i * ldc + j;
            if (beta != 0.0) {
                p0 = _mm256_fmadd_pd(bv, _mm256_loadu_pd(out), p0);
                p1 = _mm256_fmadd_pd(bv, _mm256_loadu_pd(out + 4), p1);
            }
            _mm256_storeu_pd(out, p0);
            _mm256_storeu_pd(out + 4, p1);
        }
        for (; j < n; ++j) {
            double sum = 0.0;
            if (alpha != 0.0) {
                for (size_t p = 0; p < k; ++p) sum += a[i * lda + p] * b[p * ldb + j];
            }
            double *out = c + i * ldc + j;
            *out = (k == 0 ? 0.0 : alpha * sum) + (beta == 0.0 ? 0.0 : beta * *out);
        }
    }
}
