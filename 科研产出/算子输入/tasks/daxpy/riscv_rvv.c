#include "opbench.h"
#include <riscv_vector.h>

void op_daxpy(size_t n, double alpha, const double *x, ptrdiff_t incx,
              double *y, ptrdiff_t incy) {
    if (alpha == 0.0) return;
    if (incx != 1 || incy != 1) {
        for (size_t i = 0; i < n; ++i) y[(ptrdiff_t)i * incy] += alpha * x[(ptrdiff_t)i * incx];
        return;
    }
    for (size_t i = 0; i < n;) {
        const size_t vl = __riscv_vsetvl_e64m2(n - i);
        const vfloat64m2_t xv = __riscv_vle64_v_f64m2(x + i, vl);
        vfloat64m2_t yv = __riscv_vle64_v_f64m2(y + i, vl);
        yv = __riscv_vfmacc_vf_f64m2(yv, alpha, xv, vl);
        __riscv_vse64_v_f64m2(y + i, yv, vl);
        i += vl;
    }
}
