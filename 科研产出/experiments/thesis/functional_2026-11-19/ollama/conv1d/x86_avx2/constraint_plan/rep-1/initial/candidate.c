#include "opbench.h"

void op_conv1d(size_t n, size_t k, const double *x, const double *w, double *y) {
    if (k == 0 || k > n) return;
    const size_t len = n - k + 1;

    // Vectorize the inner loop using SIMD instructions
    #pragma omp simd reduction(+:sum)
    for (size_t i = 0; i < len; ++i) {
        double sum = 0.0;
        for (size_t j = 0; j < k; ++j) sum += x[i + j] * w[k - 1 - j];
        y[i] = sum;
    }
}
