#include "opbench.h"
#include <immintrin.h>

void op_dscal(size_t n, double alpha, double *x, ptrdiff_t incx) {
    if (incx != 1) {
        for (size_t i = 0; i < n; ++i) x[(ptrdiff_t)i * incx] *= alpha;
        return;
    }
    size_t i = 0;
    const __m256d factor = _mm256_set1_pd(alpha);
    for (; n - i >= 4; i += 4)
        _mm256_storeu_pd(x + i, _mm256_mul_pd(_mm256_loadu_pd(x + i), factor));
    for (; i < n; ++i) x[i] *= alpha;
}
