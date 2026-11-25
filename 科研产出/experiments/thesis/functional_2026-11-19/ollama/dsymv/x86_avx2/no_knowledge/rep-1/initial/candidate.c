#include "opbench.h"

void op_dsymv(char uplo, size_t n, double alpha, const double *a, size_t lda,
              const double *x, ptrdiff_t incx, double beta, double *y, ptrdiff_t incy) {
    if (n == 0 || (alpha == 0.0 && beta == 1.0)) return;

    #pragma omp parallel for reduction(+:y[:n])
    for (size_t i = 0; i < n; ++i) {
        double sum = 0.0;
        if (alpha != 0.0) {
            for (size_t j = 0; j < n; ++j) {
                const int stored = uplo == 'U' ? i <= j : i >= j;
                const double aij = stored ? a[i * lda + j] : a[j * lda + i];
                sum += aij * x[(ptrdiff_t)j * incx];
            }
        }
        y[(ptrdiff_t)i * incy] = alpha * sum + (beta == 0.0 ? 0.0 : beta * y[(ptrdiff_t)i * incy]);
    }
}
