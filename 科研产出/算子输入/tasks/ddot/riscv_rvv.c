#include "opbench.h"
#include <riscv_vector.h>

double op_ddot(size_t n, const double *x, ptrdiff_t incx,
               const double *y, ptrdiff_t incy) {
    if (n == 0) return 0.0;
    if (incx != 1 || incy != 1) {
        double sum = 0.0;
        for (size_t i = 0; i < n; ++i) sum += x[(ptrdiff_t)i * incx] * y[(ptrdiff_t)i * incy];
        return sum;
    }
    const size_t full = __riscv_vsetvlmax_e64m2();
    vfloat64m2_t partial = __riscv_vfmv_v_f_f64m2(0.0, full);
    for (size_t i = 0; i < n;) {
        const size_t vl = __riscv_vsetvl_e64m2(n - i);
        const vfloat64m2_t xv = __riscv_vle64_v_f64m2(x + i, vl);
        const vfloat64m2_t yv = __riscv_vle64_v_f64m2(y + i, vl);
        partial = __riscv_vfmacc_vv_f64m2_tu(partial, xv, yv, vl);
        i += vl;
    }
    const vfloat64m1_t zero = __riscv_vfmv_v_f_f64m1(0.0, 1);
    const vfloat64m1_t sum = __riscv_vfredusum_vs_f64m2_f64m1(partial, zero, full);
    return __riscv_vfmv_f_s_f64m1_f64(sum);
}
