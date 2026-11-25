#include "opbench.h"

void op_daxpy(size_t n, double alpha, const double *x, ptrdiff_t incx,
              double *y, ptrdiff_t incy) {
    if (alpha == 0.0) return;

    // Specialize for incx == 1 to avoid index multiplication
    if (incx == 1) {
        size_t i;
        for (i = 0; i + 4 <= n; i += 4) {
            y[i] += alpha * x[i];
            y[i + 1] += alpha * x[i + 1];
            y[i + 2] += alpha * x[i + 2];
            y[i + 3] += alpha * x[i + 3];
        }
        for (; i < n; ++i) {
            y[i] += alpha * x[i];
        }
    } else if (incy == 1) {
        size_t i;
        for (i = 0; i + 4 <= n; i += 4) {
            y[i] += alpha * x[(ptrdiff_t)i * incx];
            y[i + 1] += alpha * x[(ptrdiff_t)(i + 1) * incx];
            y[i + 2] += alpha * x[(ptrdiff_t)(i + 2) * incx];
            y[i + 3] += alpha * x[(ptrdiff_t)(i + 3) * incx];
        }
        for (; i < n; ++i) {
            y[i] += alpha * x[(ptrdiff_t)i * incx];
        }
    } else {
        size_t i;
        for (i = 0; i + 4 <= n; i += 4) {
            y[(ptrdiff_t)i * incy] += alpha * x[(ptrdiff_t)i * incx];
            y[(ptrdiff_t)(i + 1) * incy] += alpha * x[(ptrdiff_t)(i + 1) * incx];
            y[(ptrdiff_t)(i + 2) * incy] += alpha * x[(ptrdiff_t)(i + 2) * incx];
            y[(ptrdiff_t)(i + 3) * incy] += alpha * x[(ptrdiff_t)(i + 3) * incx];
        }
        for (; i < n; ++i) {
            y[(ptrdiff_t)i * incy] += alpha * x[(ptrdiff_t)i * incx];
        }
    }
}
