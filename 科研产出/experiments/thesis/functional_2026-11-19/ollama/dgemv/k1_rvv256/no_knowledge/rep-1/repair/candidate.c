#include "opbench.h"

void op_dgemv(size_t m, size_t n, double alpha, const double *a, size_t lda,
              const double *x, ptrdiff_t incx, double beta, double *y, ptrdiff_t incy) {
    if (m == 0 || (n == 0 && alpha == 0.0)) return;

    for (size_t i = 0; i < m; ++i) {
        const ptrdiff_t out = (ptrdiff_t)i * incy;
        double sum = 0.0;
        if (alpha != 0.0) {
            for (size_t j = 0; j < n; ++j) sum += a[i * lda + j] * x[(ptrdiff_t)j * incx];
        }
        const double product = alpha * sum;
        y[out] = product + (beta == 0.0 ? 0.0 : beta * y[out]);
    }
}
