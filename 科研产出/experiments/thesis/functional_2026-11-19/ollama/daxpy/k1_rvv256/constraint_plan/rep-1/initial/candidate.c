#include "opbench.h"

void op_daxpy(size_t n, double alpha, const double *x, ptrdiff_t incx,
              double *y, ptrdiff_t incy) {
    if (alpha == 0.0) return;

    // Specialize for incx != 1 to eliminate index multiplication
    if (incx != 1) {
        for (size_t i = 0; i < n; ++i) {
            y[(ptrdiff_t)i * incy] += alpha * x[(ptrdiff_t)i * incx];
        }
    } else {
        // Use SIMD vectorization with unrolling
        size_t i;
        const size_t unroll_factor = 4;
        for (i = 0; i + unroll_factor <= n; i += unroll_factor) {
            y[i] += alpha * x[i];
            y[i + 1] += alpha * x[i + 1];
            y[i + 2] += alpha * x[i + 2];
            y[i + 3] += alpha * x[i + 3];
        }
        // Handle the tail
        for (; i < n; ++i) {
            y[i] += alpha * x[i];
        }
    }
}
