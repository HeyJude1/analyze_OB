#include "opbench.h"

void op_daxpy(size_t n, double alpha, const double *x, ptrdiff_t incx,
              double *y, ptrdiff_t incy) {
    if (alpha == 0.0) return;
    for (size_t i = 0; i < n; ++i) {
        y[(ptrdiff_t)i * incy] += alpha * x[(ptrdiff_t)i * incx];
    }
}
