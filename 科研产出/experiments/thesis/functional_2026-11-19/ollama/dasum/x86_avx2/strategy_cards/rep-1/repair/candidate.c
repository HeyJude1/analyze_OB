#include "opbench.h"

#include <math.h>
#include <immintrin.h>

double op_dasum(size_t n, const double *x, ptrdiff_t incx) {
    if (n == 0) return 0.0;

    size_t i = 0;
    double sum = 0.0;

    // Process elements in blocks of 4 for SIMD
    while (i + 3 < n) {
        __m256d vsum = _mm256_setzero_pd();
        for (size_t j = 0; j < 4; ++j) {
            vsum = _mm256_add_pd(vsum, _mm256_and_pd(_mm256_loadu_pd(&x[(i + j) * incx]), _mm256_set1_pd(0.0)));
        }
        sum += _mm256_reduce_add_pd(vsum);
        i += 4;
    }

    // Process remaining elements
    while (i < n) {
        sum += fabs(x[(i++) * incx]);
    }

    return sum;
}
