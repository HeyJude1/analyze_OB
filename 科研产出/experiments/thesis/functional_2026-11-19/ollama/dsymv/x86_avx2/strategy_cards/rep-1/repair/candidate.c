#include "opbench.h"

void op_dsymv(char uplo, size_t n, double alpha, const double *a, size_t lda,
              const double *x, ptrdiff_t incx, double beta, double *y, ptrdiff_t incy) {
    if (n == 0 || (alpha == 0.0 && beta == 1.0)) return;

    // Unroll factor
    const int K_UNROLL = 4;

    for (size_t i = 0; i < n; i += K_UNROLL) {
        double sum[K_UNROLL];
        __builtin_memset(sum, 0, sizeof(sum));

        if (alpha != 0.0) {
            for (size_t j = 0; j < n; ++j) {
                const int stored = uplo == 'U' ? i <= j : i >= j;
                const double aij = stored ? a[i * lda + j] : a[j * lda + i];
                sum[0] += aij * x[(ptrdiff_t)j * incx];
            }
        }

        for (int k = 1; k < K_UNROLL; ++k) {
            if (alpha != 0.0) {
                for (size_t j = 0; j < n; ++j) {
                    const int stored = uplo == 'U' ? i + k <= j : i + k >= j;
                    const double aij = stored ? a[(i + k) * lda + j] : a[j * lda + (i + k)];
                    sum[k] += aij * x[(ptrdiff_t)j * incx];
                }
            }
        }

        for (int k = 0; k < K_UNROLL; ++k) {
            const ptrdiff_t index = (ptrdiff_t)(i + k) * incy;
            y[index] = alpha * sum[k] + (beta == 0.0 ? 0.0 : beta * y[index]);
        }
    }

    // Handle remaining elements
    for (size_t i = n - n % K_UNROLL; i < n; ++i) {
        double sum = 0.0;

        if (alpha != 0.0) {
            for (size_t j = 0; j < n; ++j) {
                const int stored = uplo == 'U' ? i <= j : i >= j;
                const double aij = stored ? a[i * lda + j] : a[j * lda + i];
                sum += aij * x[(ptrdiff_t)j * incx];
            }
        }

        const ptrdiff_t index = (ptrdiff_t)i * incy;
        y[index] = alpha * sum + (beta == 0.0 ? 0.0 : beta * y[index]);
    }
}
