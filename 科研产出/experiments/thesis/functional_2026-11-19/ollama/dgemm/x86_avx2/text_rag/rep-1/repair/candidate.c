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

    if (k % 8 == 0 && lda >= 8 && ldb >= 8) {
        __m256d valpha = _mm256_set1_pd(alpha);
        __m256d vbeta = _mm256_set1_pd(beta);

        for (size_t j = 0; j < n; j += 4) {
            for (size_t i = 0; i < m; ++i) {
                __m256d vsum = _mm256_setzero_pd();
                for (size_t p = 0; p < k; p += 8) {
                    __m256d va = _mm256_loadu_pd(&a[i * lda + p]);
                    __m256d vb = _mm256_loadu_pd(&b[p * ldb + j]);
                    vsum = _mm256_add_pd(vsum, _mm256_mul_pd(va, vb));
                }
                __m256d vproduct = _mm256_mul_pd(valpha, vsum);
                __m256d vout = _mm256_loadu_pd(&c[i * ldc + j]);
                vout = _mm256_add_pd(vout, vbeta);
                vout = _mm256_add_pd(vout, vproduct);
                _mm256_storeu_pd(&c[i * ldc + j], vout);
            }
        }

        if (n % 4 != 0) {
            size_t j = n - n % 4;
            for (size_t i = 0; i < m; ++i) {
                double sum = 0.0;
                for (size_t p = 0; p < k; ++p) sum += a[i * lda + p] * b[p * ldb + j];
                const size_t out = i * ldc + j;
                const double product = alpha * sum;
                c[out] = product + beta * c[out];
            }
        }
    } else {
        for (size_t j = 0; j < n; ++j) {
            for (size_t i = 0; i < m; ++i) {
                double sum = 0.0;
                if (alpha != 0.0) {
                    for (size_t p = 0; p < k; ++p) sum += a[i * lda + p] * b[p * ldb + j];
                }
                const size_t out = i * ldc + j;
                const double product = alpha * sum;
                c[out] = product + beta * c[out];
            }
        }
    }
}
