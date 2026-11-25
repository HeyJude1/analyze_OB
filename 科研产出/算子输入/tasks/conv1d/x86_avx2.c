#include "opbench.h"
#include <immintrin.h>

void op_conv1d(size_t n, size_t k, const double *x, const double *w, double *y) {
    if (k == 0 || k > n) return;
    const size_t len = n - k + 1;
    size_t i = 0;
    for (; len - i >= 4; i += 4) {
        __m256d acc = _mm256_setzero_pd();
        for (size_t j = 0; j < k; ++j) {
            acc = _mm256_fmadd_pd(_mm256_loadu_pd(x + i + j), _mm256_set1_pd(w[k - 1 - j]), acc);
        }
        _mm256_storeu_pd(y + i, acc);
    }
    for (; i < len; ++i) {
        double sum = 0.0;
        for (size_t j = 0; j < k; ++j) sum += x[i + j] * w[k - 1 - j];
        y[i] = sum;
    }
}
