#include "opbench.h"
#include <riscv_vector.h>

void op_dgemv(size_t m, size_t n, double alpha, const double *a, size_t lda,
              const double *x, ptrdiff_t incx, double beta, double *y, ptrdiff_t incy) {
    for (size_t i = 0; i < m; ++i) {
        const ptrdiff_t out = (ptrdiff_t)i * incy;
        double sum = 0.0;
        if (n != 0 && alpha != 0.0) {
            const double *row = a + i * lda;
            if (incx == 1) {
                const size_t full = __riscv_vsetvlmax_e64m2();
                vfloat64m2_t acc = __riscv_vfmv_v_f_f64m2(0.0, full);
                for (size_t j = 0; j < n;) {
                    const size_t vl = __riscv_vsetvl_e64m2(n - j);
                    const vfloat64m2_t av = __riscv_vle64_v_f64m2(row + j, vl);
                    const vfloat64m2_t xv = __riscv_vle64_v_f64m2(x + j, vl);
                    acc = __riscv_vfmacc_vv_f64m2_tu(acc, av, xv, vl);
                    j += vl;
                }
                const vfloat64m1_t zero = __riscv_vfmv_v_f_f64m1(0.0, 1);
                const vfloat64m1_t total = __riscv_vfredusum_vs_f64m2_f64m1(acc, zero, full);
                sum = __riscv_vfmv_f_s_f64m1_f64(total);
            } else {
                for (size_t j = 0; j < n; ++j) sum += row[j] * x[(ptrdiff_t)j * incx];
            }
        }
        y[out] = (n == 0 ? 0.0 : alpha * sum) + (beta == 0.0 ? 0.0 : beta * y[out]);
    }
}
