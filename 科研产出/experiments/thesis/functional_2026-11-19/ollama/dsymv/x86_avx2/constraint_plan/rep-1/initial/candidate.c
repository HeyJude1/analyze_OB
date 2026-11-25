#include "opbench.h"

void op_dsymv(char uplo, size_t n, double alpha, const double *a, size_t lda,
              const double *x, ptrdiff_t incx, double beta, double *y, ptrdiff_t incy) {
    if (n == 0 || (alpha == 0.0 && beta == 1.0)) return;

    // Unroll factor
    const int K_UNROLL = 4;

    for (size_t i = 0; i < n; i += K_UNROLL) {
        double sum[K_UNROLL] = {0.0};

        if (alpha != 0.0) {
            for (size_t j = 0; j < n; ++j) {
                const int stored = uplo == 'U' ? i <= j : i >= j;
                const double aij = stored ? a[i * lda + j] : a[j * lda + i];
                for (int k = 0; k < K_UNROLL && i + k < n; ++k) {
                    sum[k] += aij * x[(ptrdiff_t)(j * incx + k * incx)];
                }
            }
        }

        for (int k = 0; k < K_UNROLL && i + k < n; ++k) {
            const ptrdiff_t index = (ptrdiff_t)(i + k) * incy;
            y[index] = alpha * sum[k] + (beta == 0.0 ? 0.0 : beta * y[index]);
        }
    }
}
