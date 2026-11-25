#include "opbench.h"

void op_dscal(size_t n, double alpha, double *x, ptrdiff_t incx) {
    if (incx == 1) {
        // Vectorize the loop for better performance on modern CPUs
        size_t i;
        for (i = 0; i < n / 4; ++i) {
            __m256d vec_x = _mm256_loadu_pd(&x[4 * i]);
            vec_x = _mm256_mul_pd(vec_x, _mm256_set1_pd(alpha));
            _mm256_storeu_pd(&x[4 * i], vec_x);
        }
        // Handle the remaining elements
        for (; i < n; ++i) {
            x[i] *= alpha;
        }
    } else {
        // General case with non-unit stride
        size_t i;
        for (i = 0; i < n / 4; ++i) {
            __m256d vec_x = _mm256_loadu_pd(&x[(ptrdiff_t)i * incx]);
            vec_x = _mm256_mul_pd(vec_x, _mm256_set1_pd(alpha));
            _mm256_storeu_pd(&x[(ptrdiff_t)i * incx], vec_x);
        }
        // Handle the remaining elements
        for (; i < n; ++i) {
            x[(ptrdiff_t)i * incx] *= alpha;
        }
    }
}
