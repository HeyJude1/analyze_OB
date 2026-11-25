#include "opbench.h"

#include <math.h>
#include <immintrin.h>

#define BLOCK_SIZE 16

double op_dasum(size_t n, const double *x, ptrdiff_t incx) {
    double sum = 0.0;
    size_t i;

    // Handle the main loop in blocks
    for (i = 0; i <= n - BLOCK_SIZE; i += BLOCK_SIZE) {
        __m256d vsum = _mm256_setzero_pd();
        for (int j = 0; j < BLOCK_SIZE; ++j) {
            vsum = _mm256_add_pd(vsum, _mm256_and_pd(_mm256_loadu_pd(&x[(i + j) * incx]), _mm256_set1_pd(0.0)));
        }
        double s[4];
        _mm256_storeu_pd(s, vsum);
        sum += s[0] + s[1] + s[2] + s[3];
    }

    // Handle the tail
    for (; i < n; ++i) {
        sum += fabs(x[(ptrdiff_t)i * incx]);
    }

    return sum;
}
