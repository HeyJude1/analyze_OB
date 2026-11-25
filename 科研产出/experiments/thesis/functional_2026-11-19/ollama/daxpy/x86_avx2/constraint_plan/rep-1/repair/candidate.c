#include "opbench.h"

void op_daxpy(size_t n, double alpha, const double *x, ptrdiff_t incx,
              double *y, ptrdiff_t incy) {
    if (alpha == 0.0) return;

    // Specialize for incx != 1 to eliminate index multiplication
    if (incx == 1) {
        size_t i = 0;
        __m256d v_alpha = _mm256_set1_pd(alpha);
        while (i + 4 <= n) {
            __m256d vx = _mm256_loadu_pd(&x[i]);
            __m256d vy = _mm256_loadu_pd(&y[i]);
            vy = _mm256_add_pd(vy, _mm256_mul_pd(v_alpha, vx));
            _mm256_storeu_pd(&y[i], vy);
            i += 4;
        }
    } else {
        for (size_t i = 0; i < n; ++i) {
            y[(ptrdiff_t)i * incy] += alpha * x[(ptrdiff_t)i * incx];
        }
    }

    // Handle the tail
    size_t remainder = n % 4;
    for (size_t i = n - remainder; i < n; ++i) {
        y[(ptrdiff_t)i * incy] += alpha * x[(ptrdiff_t)i * incx];
    }
}
