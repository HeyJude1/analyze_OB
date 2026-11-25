#include "opbench.h"

void op_dsymv(char uplo, size_t n, double alpha, const double *a, size_t lda,
              const double *x, ptrdiff_t incx, double beta, double *y, ptrdiff_t incy) {
    if (n == 0 || (alpha == 0.0 && beta == 1.0)) return;

    // Loop unroll factor
    const int K_UNROLL = 4;
    size_t i, j;

    for (i = 0; i < n - K_UNROLL + 1; i += K_UNROLL) {
        double sum[K_UNROLL] = {0.0};

        if (alpha != 0.0) {
            for (j = 0; j < n; ++j) {
                const int stored = uplo == 'U' ? i <= j : i >= j;
                const double aij = stored ? a[i * lda + j] : a[j * lda + i];
                sum[0] += aij * x[(ptrdiff_t)j * incx];
                sum[1] += aij * x[(ptrdiff_t)(j + 1) * incx];
                sum[2] += aij * x[(ptrdiff_t)(j + 2) * incx];
                sum[3] += aij * x[(ptrdiff_t)(j + 3) * incx];
            }
        }

        for (int k = 0; k < K_UNROLL; ++k) {
            const ptrdiff_t index = (ptrdiff_t)(i + k) * incy;
            y[index] = alpha * sum[k] + (beta == 0.0 ? 0.0 : beta * y[index]);
        }
    }

    // Handle remaining elements
    for (; i < n; ++i) {
        double sum = 0.0;

        if (alpha != 0.0) {
            for (j = 0; j < n; ++j) {
                const int stored = uplo == 'U' ? i <= j : i >= j;
                const double aij = stored ? a[i * lda + j] : a[j * lda + i];
                sum += aij * x[(ptrdiff_t)j * incx];
            }
        }

        const ptrdiff_t index = (ptrdiff_t)i * incy;
        y[index] = alpha * sum + (beta == 0.0 ? 0.0 : beta * y[index]);
    }
}
