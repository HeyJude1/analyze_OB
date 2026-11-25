#include "opbench.h"

void op_dsymv(char uplo, size_t n, double alpha, const double *a, size_t lda,
              const double *x, ptrdiff_t incx, double beta, double *y, ptrdiff_t incy) {
    if (n == 0 || (alpha == 0.0 && beta == 1.0)) return;

    // Use vector registers for better performance
    __m256d v_alpha = _mm256_set1_pd(alpha);
    __m256d v_beta = _mm256_set1_pd(beta);

    for (size_t i = 0; i < n; i += 4) {
        double sum[4] = {0.0, 0.0, 0.0, 0.0};
        __m256d v_sum = _mm256_setzero_pd();

        for (size_t j = 0; j < n; ++j) {
            const int stored = uplo == 'U' ? i <= j : i >= j;
            const double aij = stored ? a[i * lda + j] : a[j * lda + i];
            __m256d v_aij = _mm256_set1_pd(aij);
            __m256d v_x_j = _mm256_loadu_pd(&x[(ptrdiff_t)j * incx]);
            v_sum = _mm256_add_pd(v_sum, _mm256_mul_pd(v_aij, v_x_j));
        }

        for (int k = 0; k < 4; ++k) {
            sum[k] += v_sum.m256d_f32[2 * k] + v_sum.m256d_f32[2 * k + 1];
        }

        __m256d v_y_i = _mm256_loadu_pd(&y[(ptrdiff_t)i * incy]);
        v_y_i = _mm256_add_pd(_mm256_mul_pd(v_alpha, _mm256_set1_pd(sum[0])), _mm256_mul_pd(v_beta, v_y_i));
        _mm256_storeu_pd(&y[(ptrdiff_t)i * incy], v_y_i);
    }

    // Handle remaining elements
    for (size_t i = n - n % 4; i < n; ++i) {
        double sum = 0.0;
        if (alpha != 0.0) {
            for (size_t j = 0; j < n; ++j) {
                const int stored = uplo == 'U' ? i <= j : i >= j;
                const double aij = stored ? a[i * lda + j] : a[j * lda + i];
                sum += aij * x[(ptrdiff_t)j * incx];
            }
        }
        const ptrdiff_t index = (ptrdiff_t)i * incy;
        y[index] = alpha * sum + (beta == 0.0 ? 0.0 : beta * y[index]);
    }
}
