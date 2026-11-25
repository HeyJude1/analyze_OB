#include "opbench.h"
#include <riscv_vector.h>

void op_dscal(size_t n, double alpha, double *x, ptrdiff_t incx) {
    if (incx != 1) {
        for (size_t i = 0; i < n; ++i) x[(ptrdiff_t)i * incx] *= alpha;
        return;
    }
    for (size_t i = 0; i < n;) {
        const size_t vl = __riscv_vsetvl_e64m2(n - i);
        const vfloat64m2_t value = __riscv_vle64_v_f64m2(x + i, vl);
        __riscv_vse64_v_f64m2(x + i, __riscv_vfmul_vf_f64m2(value, alpha, vl), vl);
        i += vl;
    }
}
