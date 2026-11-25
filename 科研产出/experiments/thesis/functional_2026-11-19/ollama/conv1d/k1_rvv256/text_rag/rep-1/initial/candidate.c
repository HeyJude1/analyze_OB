#include "opbench.h"

void op_conv1d(size_t n, size_t k, const double *x, const double *w, double *y) {
    if (k == 0 || k > n) return;
    const size_t len = n - k + 1;

    // Register blocking
    const size_t reg_block_m = 8; // Example value, adjust based on actual register availability
    const size_t reg_block_n = 4; // Example value, adjust based on actual register availability

    for (size_t i = 0; i < len; i += reg_block_m) {
        for (size_t j = 0; j < k; j++) {
            double sum = 0.0;
            for (size_t m = 0; m < reg_block_m && i + m < len; ++m) {
                sum += x[i + m] * w[k - 1 - j];
            }
            y[i] += sum;
        }
    }

    // Tail handling
    for (size_t i = len - reg_block_m; i < len; ++i) {
        double sum = 0.0;
        for (size_t j = 0; j < k; ++j) {
            sum += x[i + j] * w[k - 1 - j];
        }
        y[i] += sum;
    }
}
