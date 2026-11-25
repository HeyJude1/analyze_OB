#include "opbench.h"

#include <math.h>

double op_dasum(size_t n, const double *x, ptrdiff_t incx) {
    if (n == 0) return 0.0;

    size_t i = 0;
    double sum = 0.0;

    // Vectorize the loop
    for (; i <= n - 4; i += 4) {
        __m256d v_x1 = _mm256_loadu_pd(&x[i]);
        __m256d v_x2 = _mm256_loadu_pd(&x[i + 2]);
        __m256d v_abs1 = _mm256_andnot_pd(_mm256_set1_pd(-0.0), v_x1);
        __m256d v_abs2 = _mm256_andnot_pd(_mm256_set1_pd(-0.0), v_x2);
        sum += _mm256_reduce_add_pd(v_abs1) + _mm256_reduce_add_pd(v_abs2);
    }

    // Handle the tail
    for (; i < n; ++i) {
        sum += fabs(x[i * incx]);
    }

    return sum;
}
