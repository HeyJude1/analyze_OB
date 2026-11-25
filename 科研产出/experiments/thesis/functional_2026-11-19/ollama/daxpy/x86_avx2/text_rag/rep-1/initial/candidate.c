#include "opbench.h"

void op_daxpy(size_t n, double alpha, const double *x, ptrdiff_t incx,
              double *y, ptrdiff_t incy) {
    if (alpha == 0.0) return;

    // Use SIMD for vectorization
    #ifdef __AVX2__
        #include <immintrin.h>
        __m256d alpha_vec = _mm256_set1_pd(alpha);
        size_t i;
        for (i = 0; i + 4 <= n; i += 4) {
            __m256d x_vec = _mm256_loadu_pd(&x[i * incx]);
            __m256d y_vec = _mm256_loadu_pd(&y[i * incy]);
            y_vec = _mm256_add_pd(y_vec, _mm256_mul_pd(alpha_vec, x_vec));
            _mm256_storeu_pd(&y[i * incy], y_vec);
        }
    #endif

    // Handle remaining elements
    for (; i < n; ++i) {
        y[(ptrdiff_t)i * incy] += alpha * x[(ptrdiff_t)i * incx];
    }
}
