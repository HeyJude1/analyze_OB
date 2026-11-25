#include "opbench.h"

void op_conv1d(size_t n, size_t k, const double *x, const double *w, double *y) {
    if (k == 0 || k > n) return;
    const size_t len = n - k + 1;

    // Vectorization parameters
    const int unroll_factor = 4;
    const int vector_elements_per_register = 4;
    const int reserved_vector_registers = 4;

    // Tail handling parameters
    const int block_size = 8;

    // Loop over the output array in blocks
    for (size_t i = 0; i < len; i += block_size) {
        size_t end = (i + block_size > len) ? len : i + block_size;
        double sum[block_size] = {0.0};

        // Vectorized inner loop
        for (size_t j = 0; j < k; ++j) {
            for (size_t ii = i; ii < end; ++ii) {
                sum[ii - i] += x[ii + j] * w[k - 1 - j];
            }
        }

        // Store the results
        for (size_t ii = i; ii < end; ++ii) {
            y[ii] = sum[ii - i];
        }
    }
}
