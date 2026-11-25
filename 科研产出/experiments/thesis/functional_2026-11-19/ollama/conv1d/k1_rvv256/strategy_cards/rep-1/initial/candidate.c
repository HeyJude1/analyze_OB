#include "opbench.h"

void op_conv1d(size_t n, size_t k, const double *x, const double *w, double *y) {
    if (k == 0 || k > n) return;
    const size_t len = n - k + 1;

    // Vectorize the convolution with SIMD instructions
    #ifdef __riscv_vector
        typedef vfloat64m8_t vec_t;
        const int vl = vsetvl_e64m8(len);
        vec_t vx, vw, vy;
        double sum[vl];

        for (size_t i = 0; i < len; i += vl) {
            vx = vle64_v_f64m8(x + i, vl);
            vy = vle64_v_f64m8(y + i, vl);

            for (int j = 0; j < k; ++j) {
                vw = vle64_v_f64m8(w + k - 1 - j, vl);
                vy = vfma_vv_f64(vy, vx, vw, vl);
            }

            vse64_v_f64m8(y + i, vy, vl);
        }
    #else
        for (size_t i = 0; i < len; ++i) {
            double sum = 0.0;
            for (size_t j = 0; j < k; ++j) sum += x[i + j] * w[k - 1 - j];
            y[i] = sum;
        }
    #endif
}
