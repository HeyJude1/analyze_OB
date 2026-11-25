#include "opbench.h"

double op_ddot(size_t n, const double *x, ptrdiff_t incx,
               const double *y, ptrdiff_t incy) {
    double sum = 0.0;
    if (incx == 1 && incy == 1) {
        for (size_t i = 0; i + 8 <= n; i += 8) {
            __builtin_prefetch(&x[i + 64], 0, 3);
            sum += x[i] * y[i];
            sum += x[i + 1] * y[i + 1];
            sum += x[i + 2] * y[i + 2];
            sum += x[i + 3] * y[i + 3];
            sum += x[i + 4] * y[i + 4];
            sum += x[i + 5] * y[i + 5];
            sum += x[i + 6] * y[i + 6];
            sum += x[i + 7] * y[i + 7];
        }
        for (; i < n; ++i) {
            sum += x[i] * y[i];
        }
    } else {
        for (size_t i = 0; i + 8 <= n; i += 8) {
            __builtin_prefetch(&x[(ptrdiff_t)i * incx], 0, 3);
            sum += x[(ptrdiff_t)i * incx] * y[(ptrdiff_t)i * incy];
            sum += x[(ptrdiff_t)(i + 1) * incx] * y[(ptrdiff_t)(i + 1) * incy];
            sum += x[(ptrdiff_t)(i + 2) * incx] * y[(ptrdiff_t)(i + 2) * incy];
            sum += x[(ptrdiff_t)(i + 3) * incx] * y[(ptrdiff_t)(i + 3) * incy];
            sum += x[(ptrdiff_t)(i + 4) * incx] * y[(ptrdiff_t)(i + 4) * incy];
            sum += x[(ptrdiff_t)(i + 5) * incx] * y[(ptrdiff_t)(i + 5) * incy];
            sum += x[(ptrdiff_t)(i + 6) * incx] * y[(ptrdiff_t)(i + 6) * incy];
            sum += x[(ptrdiff_t)(i + 7) * incx] * y[(ptrdiff_t)(i + 7) * incy];
        }
        for (; i < n; ++i) {
            sum += x[(ptrdiff_t)i * incx] * y[(ptrdiff_t)i * incy];
        }
    }
    return sum;
}
