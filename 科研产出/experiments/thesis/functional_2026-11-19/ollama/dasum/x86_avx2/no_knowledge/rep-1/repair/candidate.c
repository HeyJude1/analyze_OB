#include "opbench.h"
#include <immintrin.h>
#include <math.h>

double op_dasum(size_t n, const double *x, ptrdiff_t incx) {
    if (n == 0) return 0.0;

    double sum = 0.0;
    size_t i = 0;

    // Process elements in chunks of 4 using AVX2
    for (; i <= n - 8; i += 8) {
        __m256d vec_x1 = _mm256_loadu_pd(&x[i]);
        __m256d vec_x2 = _mm256_loadu_pd(&x[i + 4]);
        __m256d abs_vec_x1 = _mm256_andnot_pd(_mm256_set1_pd(-0.0), vec_x1);
        __m256d abs_vec_x2 = _mm256_andnot_pd(_mm256_set1_pd(-0.0), vec_x2);
        sum += _mm256_reduce_add_pd(abs_vec_x1) + _mm256_reduce_add_pd(abs_vec_x2);
    }

    // Process remaining elements
    for (; i < n; ++i) {
        sum += fabs(x[i * incx]);
    }

    return sum;
}
