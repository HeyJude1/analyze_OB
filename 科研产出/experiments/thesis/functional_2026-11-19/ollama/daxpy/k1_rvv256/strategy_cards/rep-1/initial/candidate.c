#include "opbench.h"

void op_daxpy(size_t n, double alpha, const double *x, ptrdiff_t incx,
              double *y, ptrdiff_t incy) {
    if (alpha == 0.0) return;

    // Specialize for incx == 1 to avoid index multiplication
    if (incx == 1) {
        for (size_t i = 0; i + 4 <= n; i += 4) {
            y[i] += alpha * x[i];
            y[i + 1] += alpha * x[i + 1];
            y[i + 2] += alpha * x[i + 2];
            y[i + 3] += alpha * x[i + 3];
        }
        for (; i < n; ++i) {
            y[i] += alpha * x[i];
        }
    } else {
        for (size_t i = 0; i < n; ++i) {
            y[(ptrdiff_t)i * incy] += alpha * x[(ptrdiff_t)i * incx];
        }
    }
}
