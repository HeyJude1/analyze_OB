#include "opbench.h"
#include <riscv_vector.h>

static double stored_dot(const double *row, const double *x, ptrdiff_t incx, size_t start, size_t end) {
    if (incx != 1) {
        double sum = 0.0;
        for (size_t j = start; j < end; ++j) sum += row[j] * x[(ptrdiff_t)j * incx];
        return sum;
    }
    const size_t full = __riscv_vsetvlmax_e64m2();
    vfloat64m2_t acc = __riscv_vfmv_v_f_f64m2(0.0, full);
    for (size_t j = start; j < end;) {
        const size_t vl = __riscv_vsetvl_e64m2(end - j);
        const vfloat64m2_t av = __riscv_vle64_v_f64m2(row + j, vl);
        const vfloat64m2_t xv = __riscv_vle64_v_f64m2(x + j, vl);
        acc = __riscv_vfmacc_vv_f64m2_tu(acc, av, xv, vl);
        j += vl;
    }
    const vfloat64m1_t zero = __riscv_vfmv_v_f_f64m1(0.0, 1);
    return __riscv_vfmv_f_s_f64m1_f64(__riscv_vfredusum_vs_f64m2_f64m1(acc, zero, full));
}

void op_dsymv(char uplo, size_t n, double alpha, const double *a, size_t lda,
              const double *x, ptrdiff_t incx, double beta, double *y, ptrdiff_t incy) {
    if (n == 0 || (alpha == 0.0 && beta == 1.0)) return;
    for (size_t i = 0; i < n; ++i) {
        double sum = 0.0;
        if (alpha != 0.0) {
            const size_t start = uplo == 'U' ? i : 0;
            const size_t end = uplo == 'U' ? n : i + 1;
            sum = stored_dot(a + i * lda, x, incx, start, end);
            const size_t mirrored_begin = uplo == 'U' ? 0 : i + 1;
            const size_t mirrored_end = uplo == 'U' ? i : n;
            for (size_t j = mirrored_begin; j < mirrored_end; ++j)
                sum += a[j * lda + i] * x[(ptrdiff_t)j * incx];
        }
        const ptrdiff_t out = (ptrdiff_t)i * incy;
        y[out] = alpha * sum + (beta == 0.0 ? 0.0 : beta * y[out]);
    }
}
