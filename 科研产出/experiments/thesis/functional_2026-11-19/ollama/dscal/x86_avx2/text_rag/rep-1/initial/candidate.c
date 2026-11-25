#include "opbench.h"

void op_dscal(size_t n, double alpha, double *x, ptrdiff_t incx) {
    if (incx == 1) {
        __m256d v_alpha = _mm256_set1_pd(alpha);
        for (size_t i = 0; i < n / 4; ++i) {
            __m256d v_x = _mm256_loadu_pd(x + 4 * i);
            v_x = _mm256_mul_pd(v_x, v_alpha);
            _mm256_storeu_pd(x + 4 * i, v_x);
        }
        for (size_t i = n / 4 * 4; i < n; ++i) {
            x[i] *= alpha;
        }
    } else {
        __m256d v_alpha = _mm256_set1_pd(alpha);
        for (size_t i = 0; i < n / 4; ++i) {
            __m256d v_x = _mm256_loadu_pd(x + i * incx);
            v_x = _mm256_mul_pd(v_x, v_alpha);
            _mm256_storeu_pd(x + i * incx, v_x);
        }
        for (size_t i = n / 4 * 4; i < n; ++i) {
            x[i * incx] *= alpha;
        }
    }
}
