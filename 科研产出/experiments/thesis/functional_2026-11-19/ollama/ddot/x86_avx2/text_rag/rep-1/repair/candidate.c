#include "opbench.h"

double op_ddot(size_t n, const double *x, ptrdiff_t incx,
               const double *y, ptrdiff_t incy) {
    double sum = 0.0;
    if (incx == 1 && incy == 1) {
        for (size_t i = 0; i + 4 <= n; i += 4) {
            __m256d x_vec = _mm256_loadu_pd(&x[i]);
            __m256d y_vec = _mm256_loadu_pd(&y[i]);
            __m256d prod_vec = _mm256_mul_pd(x_vec, y_vec);
            sum += _mm256_reduce_add_pd(prod_vec);
        }
        for (; i < n; ++i) {
            sum += x[i] * y[i];
        }
    } else {
        for (size_t i = 0; i + 4 <= n; i += 4) {
            __m256d x_vec = _mm256_loadu_pd(&x[(ptrdiff_t)i * incx]);
            __m256d y_vec = _mm256_loadu_pd(&y[(ptrdiff_t)i * incy]);
            __m256d prod_vec = _mm256_mul_pd(x_vec, y_vec);
            sum += _mm256_reduce_add_pd(prod_vec);
        }
        for (; i < n; ++i) {
            sum += x[(ptrdiff_t)i * incx] * y[(ptrdiff_t)i * incy];
        }
    }
    return sum;
}
