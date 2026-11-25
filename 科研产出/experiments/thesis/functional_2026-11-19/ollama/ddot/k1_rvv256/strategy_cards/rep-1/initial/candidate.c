#include "opbench.h"

double op_ddot(size_t n, const double *x, ptrdiff_t incx,
               const double *y, ptrdiff_t incy) {
    if (incx == 1 && incy == 1) {
        double sum = 0.0;
        for (size_t i = 0; i < n; ++i) {
            sum += x[i] * y[i];
        }
        return sum;
    } else {
        double sum = 0.0;
        for (size_t i = 0; i < n; ++i) {
            sum += x[(ptrdiff_t)i * incx] * y[(ptrdiff_t)i * incy];
        }
        return sum;
    }
}
