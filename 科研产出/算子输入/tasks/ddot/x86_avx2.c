#include "opbench.h"
#include <immintrin.h>

double op_ddot(size_t n, const double *x, ptrdiff_t incx,
               const double *y, ptrdiff_t incy) {
    double sum = 0.0;
    size_t i = 0;
    if (incx == 1 && incy == 1 && n >= 8) {
        __m256d p0 = _mm256_setzero_pd();
        __m256d p1 = _mm256_setzero_pd();
        for (; n - i >= 8; i += 8) {
            p0 = _mm256_fmadd_pd(_mm256_loadu_pd(x + i), _mm256_loadu_pd(y + i), p0);
            p1 = _mm256_fmadd_pd(_mm256_loadu_pd(x + i + 4), _mm256_loadu_pd(y + i + 4), p1);
        }
        double lanes[4];
        _mm256_storeu_pd(lanes, _mm256_add_pd(p0, p1));
        sum = (lanes[0] + lanes[1]) + (lanes[2] + lanes[3]);
    }
    for (; i < n; ++i) sum += x[(ptrdiff_t)i * incx] * y[(ptrdiff_t)i * incy];
    return sum;
}
