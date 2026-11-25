#include "opbench.h"

void op_dscal(size_t n, double alpha, double *x, ptrdiff_t incx) {
    if (incx == 1) {
        // Vectorized version for contiguous memory
        size_t i = 0;
        for (; i <= n - 8; i += 8) {
            __m256d v_x = _mm256_loadu_pd(&x[i]);
            __m256d v_alpha = _mm256_set1_pd(alpha);
            v_x = _mm256_mul_pd(v_x, v_alpha);
            _mm256_storeu_pd(&x[i], v_x);
        }
        for (; i < n; ++i) {
            x[i] *= alpha;
        }
    } else {
        // Non-vectorized version for non-contiguous memory
        for (size_t i = 0; i < n; ++i) {
            x[(ptrdiff_t)i * incx] *= alpha;
        }
    }
}
