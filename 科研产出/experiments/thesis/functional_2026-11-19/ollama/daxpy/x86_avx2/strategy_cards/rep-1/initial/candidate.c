#include "opbench.h"

void op_daxpy(size_t n, double alpha, const double *x, ptrdiff_t incx,
              double *y, ptrdiff_t incy) {
    if (alpha == 0.0) return;

    // Vectorize the loop for better performance
    size_t i;
    for (i = 0; i + 4 <= n; i += 4) {
        y[i * incy] += alpha * x[i * incx];
        y[(i + 1) * incy] += alpha * x[(i + 1) * incx];
        y[(i + 2) * incy] += alpha * x[(i + 2) * incx];
        y[(i + 3) * incy] += alpha * x[(i + 3) * incx];
    }

    // Handle the tail elements
    for (; i < n; ++i) {
        y[i * incy] += alpha * x[i * incx];
    }
}
