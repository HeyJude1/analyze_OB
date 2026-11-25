#include "opbench.h"

void op_dgemv(size_t m, size_t n, double alpha, const double *a, size_t lda,
              const double *x, ptrdiff_t incx, double beta, double *y, ptrdiff_t incy) {
    if (alpha == 0.0) {
        if (beta != 0.0) {
            for (size_t i = 0; i < m; ++i) {
                const ptrdiff_t out = (ptrdiff_t)i * incy;
                y[out] *= beta;
            }
        }
        return;
    }

    if (incx == 1 && incy == 1) {
        for (size_t i = 0; i < m; ++i) {
            const ptrdiff_t out = (ptrdiff_t)i * incy;
            double sum = 0.0;
            for (size_t j = 0; j < n; ++j) {
                sum += a[i * lda + j] * x[j];
            }
            y[out] = alpha * sum + beta * y[out];
        }
    } else if (incx == 1 && incy != 1) {
        for (size_t i = 0; i < m; ++i) {
            const ptrdiff_t out = (ptrdiff_t)i * incy;
            double sum = 0.0;
            for (size_t j = 0; j < n; ++j) {
                sum += a[i * lda + j] * x[j];
            }
            y[out] = alpha * sum + beta * y[out];
        }
    } else if (incx != 1 && incy == 1) {
        for (size_t i = 0; i < m; ++i) {
            const ptrdiff_t out = (ptrdiff_t)i * incy;
            double sum = 0.0;
            for (size_t j = 0; j < n; ++j) {
                sum += a[i * lda + j] * x[j * incx];
            }
            y[out] = alpha * sum + beta * y[out];
        }
    } else if (incx != 1 && incy != 1) {
        for (size_t i = 0; i < m; ++i) {
            const ptrdiff_t out = (ptrdiff_t)i * incy;
            double sum = 0.0;
            for (size_t j = 0; j < n; ++j) {
                sum += a[i * lda + j] * x[j * incx];
            }
            y[out] = alpha * sum + beta * y[out];
        }
    }
}
