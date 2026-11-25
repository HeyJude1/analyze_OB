#include "opbench.h"
#include <immintrin.h>
#include <math.h>

double op_dasum(size_t n, const double *x, ptrdiff_t incx) {
    if (incx != 1) {
        double result = 0.0;
        for (size_t i = 0; i < n; ++i) result += fabs(x[(ptrdiff_t)i * incx]);
        return result;
    }
    const __m256d sign = _mm256_set1_pd(-0.0);
    __m256d first = _mm256_setzero_pd(), second = _mm256_setzero_pd();
    size_t i = 0;
    for (; n - i >= 8; i += 8) {
        first = _mm256_add_pd(first, _mm256_andnot_pd(sign, _mm256_loadu_pd(x + i)));
        second = _mm256_add_pd(second, _mm256_andnot_pd(sign, _mm256_loadu_pd(x + i + 4)));
    }
    double lanes[4];
    _mm256_storeu_pd(lanes, _mm256_add_pd(first, second));
    double sum = lanes[0] + lanes[1] + lanes[2] + lanes[3];
    for (; i < n; ++i) sum += fabs(x[i]);
    return sum;
}
