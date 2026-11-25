#include "opbench.h"
#include <immintrin.h>

void op_dsymv(char uplo, size_t n, double alpha, const double *a, size_t lda,
              const double *x, ptrdiff_t incx, double beta, double *y, ptrdiff_t incy) {
    if (n == 0 || (alpha == 0.0 && beta == 1.0)) return;

    // Use vector registers for better performance
    __m512d v_alpha = _mm512_set1_pd(alpha);
    __m512d v_beta = _mm512_set1_pd(beta);

    for (size_t i = 0; i < n; i += 8) {
        double sum[8] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        __m512d v_sum = _mm512_setzero_pd();

        for (size_t j = 0; j < n; ++j) {
            const int stored = uplo == 'U' ? i <= j : i >= j;
            const double aij = stored ? a[i * lda + j] : a[j * lda + i];
            __m512d v_aij = _mm512_set1_pd(aij);
            __m512d v_x_j = _mm512_loadu_pd(&x[(ptrdiff_t)j * incx]);
            v_sum = _mm512_add_pd(v_sum, _mm512_mul_pd(v_aij, v_x_j));
        }

        for (int k = 0; k < 8; ++k) {
            sum[k] += v_sum.m512d_f32[4 * k] + v_sum.m512d_f32[4 * k + 1] +
                      v_sum.m512d_f32[4 * k + 2] + v_sum.m512d_f32[4 * k + 3];
        }

        __m512d v_y_i = _mm512_loadu_pd(&y[(ptrdiff_t)i * incy]);
        v_y_i = _mm512_add_pd(_mm512_mul_pd(v_alpha, _mm512_set1_pd(sum[0])), _mm512_mul_pd(v_beta, v_y_i));
        _mm512_storeu_pd(&y[(ptrdiff_t)i * incy], v_y_i);
    }

    // Handle remaining elements
    for (size_t i = n - n % 8; i < n; ++i) {
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
