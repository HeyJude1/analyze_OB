#include "opbench.h"

double op_ddot(size_t n, const double *x, ptrdiff_t incx,
               const double *y, ptrdiff_t incy) {
    double sum = 0.0;
    for (size_t i = 0; i < n; ++i) {
        sum += x[(ptrdiff_t)i * incx] * y[(ptrdiff_t)i * incy];
    }
    return sum;
}
