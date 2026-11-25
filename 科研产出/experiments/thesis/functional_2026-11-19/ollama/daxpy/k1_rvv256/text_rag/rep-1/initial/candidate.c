#include "opbench.h"

void op_daxpy(size_t n, double alpha, const double *x, ptrdiff_t incx,
              double *y, ptrdiff_t incy) {
    if (alpha == 0.0) return;

    // Step 1: Determine the optimal unroll factor based on cache line size
    const int UNROLL_FACTOR = 8; // Adjust this value based on actual cache line size and vector width

    // Step 2: Handle the main loop with unrolling for better performance
    for (size_t i = 0; i + UNROLL_FACTOR <= n; i += UNROLL_FACTOR) {
        y[i * incy] += alpha * x[i * incx];
        y[(i + 1) * incy] += alpha * x[(i + 1) * incx];
        y[(i + 2) * incy] += alpha * x[(i + 2) * incx];
        y[(i + 3) * incy] += alpha * x[(i + 3) * incx];
        y[(i + 4) * incy] += alpha * x[(i + 4) * incx];
        y[(i + 5) * incy] += alpha * x[(i + 5) * incx];
        y[(i + 6) * incy] += alpha * x[(i + 6) * incx];
        y[(i + 7) * incy] += alpha * x[(i + 7) * incx];
    }

    // Step 3: Handle the remaining elements
    for (size_t i = n - n % UNROLL_FACTOR; i < n; ++i) {
        y[i * incy] += alpha * x[i * incx];
    }
}
