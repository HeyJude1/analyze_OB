#include "opbench.h"
#include <riscv_vector.h>

void op_conv1d(size_t n, size_t k, const double *x, const double *w, double *y) {
    if (k == 0 || k > n) return;
    const size_t len = n - k + 1;
    for (size_t i = 0; i < len;) {
        const size_t vl = __riscv_vsetvl_e64m2(len - i);
        vfloat64m2_t acc = __riscv_vfmv_v_f_f64m2(0.0, vl);
        for (size_t j = 0; j < k; ++j) {
            acc = __riscv_vfmacc_vf_f64m2(acc, w[k - 1 - j], __riscv_vle64_v_f64m2(x + i + j, vl), vl);
        }
        __riscv_vse64_v_f64m2(y + i, acc, vl);
        i += vl;
    }
}
