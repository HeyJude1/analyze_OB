#include <immintrin.h>
#include "opbench.h"

void op_daxpy(size_t n, double alpha, const double *x, ptrdiff_t incx,
              double *y, ptrdiff_t incy) {
    if (alpha == 0.0) return;

    size_t i;
    const size_t unroll_factor = 4;
    for (i = 0; i + unroll_factor <= n; i += unroll_factor) {
        __m256d y_vec = _mm256_loadu_pd(&y[i * incy]);
        __m256d x_vec = _mm256_loadu_pd(&x[i * incx]);
        __m256d alpha_vec = _mm256_set1_pd(alpha);
        y_vec = _mm256_add_pd(y_vec, _mm256_mul_pd(x_vec, alpha_vec));
        _mm256_storeu_pd(&y[i * incy], y_vec);
    }

    for (; i < n; ++i) {
        y[(ptrdiff_t)i * incy] += alpha * x[(ptrdiff_t)i * incx];
    }
}
