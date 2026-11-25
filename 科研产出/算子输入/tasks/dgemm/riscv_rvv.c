#include "opbench.h"
#include <riscv_vector.h>

void op_dgemm(size_t m, size_t n, size_t k, double alpha, const double *a, size_t lda,
              const double *b, size_t ldb, double beta, double *c, size_t ldc) {
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n;) {
            const size_t vl = __riscv_vsetvl_e64m2(n - j);
            vfloat64m2_t acc = __riscv_vfmv_v_f_f64m2(0.0, vl);
            if (alpha != 0.0) {
                for (size_t p = 0; p < k; ++p) {
                    const vfloat64m2_t bp = __riscv_vle64_v_f64m2(b + p * ldb + j, vl);
                    acc = __riscv_vfmacc_vf_f64m2(acc, a[i * lda + p], bp, vl);
                }
            }
            if (k != 0) acc = __riscv_vfmul_vf_f64m2(acc, alpha, vl);
            double *out = c + i * ldc + j;
            if (beta != 0.0) {
                acc = __riscv_vfmacc_vf_f64m2(acc, beta, __riscv_vle64_v_f64m2(out, vl), vl);
            }
            __riscv_vse64_v_f64m2(out, acc, vl);
            j += vl;
        }
    }
}
