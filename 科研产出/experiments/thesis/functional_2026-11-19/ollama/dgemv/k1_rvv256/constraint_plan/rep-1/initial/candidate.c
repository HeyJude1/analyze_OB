#include "opbench.h"

void op_dgemv(size_t m, size_t n, double alpha, const double *a, size_t lda,
              const double *x, ptrdiff_t incx, double beta, double *y, ptrdiff_t incy) {
    if (incx == 1 && incy == 1) {
        for (size_t i = 0; i < m; ++i) {
            double sum = 0.0;
            if (alpha != 0.0) {
                for (size_t j = 0; j < n; ++j) sum += a[i * lda + j] * x[j];
            }
            const double product = alpha * sum;
            y[i] = product + beta * y[i];
        }
    } else if (incx == 1) {
        for (size_t i = 0; i < m; ++i) {
            double sum = 0.0;
            if (alpha != 0.0) {
                for (size_t j = 0; j < n; ++j) sum += a[i * lda + j] * x[j * incx];
            }
            const double product = alpha * sum;
            y[i * incy] = product + beta * y[i * incy];
        }
    } else {
        for (size_t i = 0; i < m; ++i) {
            double sum = 0.0;
            if (alpha != 0.0) {
                for (size_t j = 0; j < n; ++j) sum += a[i * lda + j] * x[(ptrdiff_t)j * incx];
            }
            const double product = alpha * sum;
            y[(ptrdiff_t)i * incy] = product + beta * y[(ptrdiff_t)i * incy];
        }
    }
}
