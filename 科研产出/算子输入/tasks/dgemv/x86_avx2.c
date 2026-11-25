#include "opbench.h"
#include <immintrin.h>

void op_dgemv(size_t m, size_t n, double alpha, const double *a, size_t lda,
              const double *x, ptrdiff_t incx, double beta, double *y, ptrdiff_t incy) {
    for (size_t i = 0; i < m; ++i) {
        const ptrdiff_t out = (ptrdiff_t)i * incy;
        double sum = 0.0;
        if (n != 0 && alpha != 0.0) {
            const double *row = a + i * lda;
            size_t j = 0;
            if (incx == 1 && n >= 8) {
                __m256d p0 = _mm256_setzero_pd();
                __m256d p1 = _mm256_setzero_pd();
                for (; n - j >= 8; j += 8) {
                    p0 = _mm256_fmadd_pd(_mm256_loadu_pd(row + j), _mm256_loadu_pd(x + j), p0);
                    p1 = _mm256_fmadd_pd(_mm256_loadu_pd(row + j + 4), _mm256_loadu_pd(x + j + 4), p1);
                }
                double lanes[4];
                _mm256_storeu_pd(lanes, _mm256_add_pd(p0, p1));
                sum = lanes[0] + lanes[1] + lanes[2] + lanes[3];
            }
            for (; j < n; ++j) sum += row[j] * x[(ptrdiff_t)j * incx];
        }
        y[out] = (n == 0 ? 0.0 : alpha * sum) + (beta == 0.0 ? 0.0 : beta * y[out]);
    }
}
