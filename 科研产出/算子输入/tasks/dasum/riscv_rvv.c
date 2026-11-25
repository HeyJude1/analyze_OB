#include "opbench.h"
#include <math.h>
#include <riscv_vector.h>

double op_dasum(size_t n, const double *x, ptrdiff_t incx) {
    if (n == 0) return 0.0;
    if (incx != 1) {
        double result = 0.0;
        for (size_t i = 0; i < n; ++i) result += fabs(x[(ptrdiff_t)i * incx]);
        return result;
    }
    const size_t full = __riscv_vsetvlmax_e64m2();
    vfloat64m2_t acc = __riscv_vfmv_v_f_f64m2(0.0, full);
    for (size_t i = 0; i < n;) {
        const size_t vl = __riscv_vsetvl_e64m2(n - i);
        const vfloat64m2_t value = __riscv_vle64_v_f64m2(x + i, vl);
        const vfloat64m2_t magnitude = __riscv_vfsgnjx_vv_f64m2(value, value, vl);
        acc = __riscv_vfadd_vv_f64m2_tu(acc, acc, magnitude, vl);
        i += vl;
    }
    const vfloat64m1_t zero = __riscv_vfmv_v_f_f64m1(0.0, 1);
    const vfloat64m1_t sum = __riscv_vfredusum_vs_f64m2_f64m1(acc, zero, full);
    return __riscv_vfmv_f_s_f64m1_f64(sum);
}
