#include "opbench.h"

void op_dsymv(char uplo, size_t n, double alpha, const double *a, size_t lda,
              const double *x, ptrdiff_t incx, double beta, double *y, ptrdiff_t incy) {
    if (n == 0 || (alpha == 0.0 && beta == 1.0)) return;

    // Vectorize the inner loop using SIMD instructions
    size_t i, j;
    for (i = 0; i < n; ++i) {
        double sum = 0.0;
        if (alpha != 0.0) {
            for (j = 0; j < n; j += 4) { // Unroll factor of 4
                const int stored1 = uplo == 'U' ? i <= j : i >= j;
                const int stored2 = uplo == 'U' ? i <= j + 1 : i >= j + 1;
                const int stored3 = uplo == 'U' ? i <= j + 2 : i >= j + 2;
                const int stored4 = uplo == 'U' ? i <= j + 3 : i >= j + 3;

                const double aij1 = stored1 ? a[i * lda + j] : a[j * lda + i];
                const double aij2 = stored2 ? a[i * lda + j + 1] : a[j + 1 * lda + i];
                const double aij3 = stored3 ? a[i * lda + j + 2] : a[j + 2 * lda + i];
                const double aij4 = stored4 ? a[i * lda + j + 3] : a[j + 3 * lda + i];

                sum += aij1 * x[(ptrdiff_t)j * incx];
                sum += aij2 * x[(ptrdiff_t)(j + 1) * incx];
                sum += aij3 * x[(ptrdiff_t)(j + 2) * incx];
                sum += aij4 * x[(ptrdiff_t)(j + 3) * incx];
            }
        }
        const ptrdiff_t index = (ptrdiff_t)i * incy;
        y[index] = alpha * sum + (beta == 0.0 ? 0.0 : beta * y[index]);
    }
}
