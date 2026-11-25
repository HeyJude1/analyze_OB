#include "opbench.h"
#include <immintrin.h>

void op_daxpy(size_t n, double alpha, const double *x, ptrdiff_t incx,
              double *y, ptrdiff_t incy) {
    if (alpha == 0.0) return;
    size_t i = 0;
    if (incx == 1 && incy == 1) {
        const __m256d factor = _mm256_set1_pd(alpha);
        for (; n - i >= 8; i += 8) {
            const __m256d x0 = _mm256_loadu_pd(x + i);
            const __m256d x1 = _mm256_loadu_pd(x + i + 4);
            const __m256d y0 = _mm256_loadu_pd(y + i);
            const __m256d y1 = _mm256_loadu_pd(y + i + 4);
            _mm256_storeu_pd(y + i, _mm256_fmadd_pd(factor, x0, y0));
            _mm256_storeu_pd(y + i + 4, _mm256_fmadd_pd(factor, x1, y1));
        }
    }
    for (; i < n; ++i) y[(ptrdiff_t)i * incy] += alpha * x[(ptrdiff_t)i * incx];
}
