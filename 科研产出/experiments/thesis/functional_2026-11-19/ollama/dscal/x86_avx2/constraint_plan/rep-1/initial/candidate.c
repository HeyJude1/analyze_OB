#include "opbench.h"

void op_dscal(size_t n, double alpha, double *x, ptrdiff_t incx) {
    if (incx == 1) {
        // Vectorize the loop for performance improvement
        size_t i = 0;
        for (; i <= n - 4; i += 4) {
            __m256d v_x0 = _mm256_loadu_pd(&x[i]);
            __m256d v_alpha = _mm256_set1_pd(alpha);
            __m256d v_result = _mm256_mul_pd(v_x0, v_alpha);
            _mm256_storeu_pd(&x[i], v_result);
        }
        // Handle the remaining elements
        for (; i < n; ++i) {
            x[i] *= alpha;
        }
    } else {
        // Non-unit stride case
        size_t i = 0;
        for (; i <= n - 4; i += 4) {
            __m256d v_x0 = _mm256_loadu_pd(&x[(ptrdiff_t)i * incx]);
            __m256d v_alpha = _mm256_set1_pd(alpha);
            __m256d v_result = _mm256_mul_pd(v_x0, v_alpha);
            _mm256_storeu_pd(&x[(ptrdiff_t)i * incx], v_result);
        }
        // Handle the remaining elements
        for (; i < n; ++i) {
            x[(ptrdiff_t)i * incx] *= alpha;
        }
    }
}
