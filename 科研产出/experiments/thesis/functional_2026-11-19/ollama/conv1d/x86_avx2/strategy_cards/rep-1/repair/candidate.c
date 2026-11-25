#include "opbench.h"

void op_conv1d(size_t n, size_t k, const double *x, const double *w, double *y) {
    if (k == 0 || k > n) return;
    const size_t len = n - k + 1;

    // Vectorize the convolution with AVX2
    __m256d v_zero = _mm256_setzero_pd();
    for (size_t i = 0; i < len; i += 4) {
        __m256d sum = v_zero;
        for (size_t j = 0; j < k; ++j) {
            __m256d vx = _mm256_loadu_pd(&x[i + j]);
            __m256d vw = _mm256_set1_pd(w[k - 1 - j]);
            sum = _mm256_add_pd(sum, _mm256_mul_pd(vx, vw));
        }
        _mm256_storeu_pd(&y[i], sum);
    }

    // Handle the tail
    for (size_t i = len & ~3; i < len; ++i) {
        double sum = 0.0;
        for (size_t j = 0; j < k; ++j) {
            sum += x[i + j] * w[k - 1 - j];
        }
        y[i] = sum;
    }
}
