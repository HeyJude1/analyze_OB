#include "opbench.h"
#include <math.h>

double op_dasum(size_t n, const double *x, ptrdiff_t incx) {
    if (n == 0) return 0.0;

    size_t i = 0;
    double sum = 0.0;

    // Process elements in blocks of 4
    for (; i <= n - 4; i += 4) {
        __m256d vsum = _mm256_setzero_pd();
        for (int j = 0; j < 4; ++j) {
            vsum = _mm256_add_pd(vsum, _mm256_and_pd(_mm256_loadu_pd(&x[i + j * incx]), _mm256_set1_pd(0.0)));
        }
        sum += _mm256_reduce_add_pd(vsum);
    }

    // Process remaining elements
    for (; i < n; ++i) {
        sum += fabs(x[(ptrdiff_t)i * incx]);
    }

    return sum;
}
