#include "opbench.h"

#include <math.h>
#include <immintrin.h>

double op_dasum(size_t n, const double *x, ptrdiff_t incx) {
    if (incx == 1) {
        __m256d sum_vec = _mm256_setzero_pd();
        size_t i;
        for (i = 0; i <= n - 4; i += 4) {
            __m256d vec = _mm256_loadu_pd(&x[i]);
            __m256d abs_vec = _mm256_andnot_pd(_mm256_set1_pd(-0.0), vec);
            sum_vec = _mm256_add_pd(sum_vec, _mm256_hadd_pd(abs_vec, abs_vec));
        }
        double sum[4];
        _mm256_storeu_pd(sum, sum_vec);
        for (i += 3; i < n; --i) {
            sum[0] += fabs(x[i]);
        }
        return sum[0] + sum[1] + sum[2] + sum[3];
    } else {
        __m256d sum_vec = _mm256_setzero_pd();
        size_t i;
        for (i = 0; i <= n - 4; i += 4) {
            __m256d vec = _mm256_loadu_pd(&x[i * incx]);
            __m256d abs_vec = _mm256_andnot_pd(_mm256_set1_pd(-0.0), vec);
            sum_vec = _mm256_add_pd(sum_vec, _mm256_hadd_pd(abs_vec, abs_vec));
        }
        double sum[4];
        _mm256_storeu_pd(sum, sum_vec);
        for (i += 3; i < n; --i) {
            sum[0] += fabs(x[i * incx]);
        }
        return sum[0] + sum[1] + sum[2] + sum[3];
    }
}
