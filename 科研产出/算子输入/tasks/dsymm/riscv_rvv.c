#include "opbench.h"
#include <riscv_vector.h>

static double symmetric_at(const double *a, size_t lda, size_t i, size_t j, char uplo) {
    const int stored = uplo == 'U' ? i <= j : i >= j;
    return stored ? a[i * lda + j] : a[j * lda + i];
}

void op_dsymm(char side, char uplo, size_t m, size_t n, double alpha,
              const double *a, size_t lda, const double *b, size_t ldb,
              double beta, double *c, size_t ldc) {
    if (m == 0 || n == 0 || (alpha == 0.0 && beta == 1.0)) return;
    if (side == 'R') {
        for (size_t i = 0; i < m; ++i) {
            for (size_t j = 0; j < n; ++j) {
                double sum = 0.0;
                if (alpha != 0.0) {
                    for (size_t p = 0; p < n; ++p)
                        sum += b[i * ldb + p] * symmetric_at(a, lda, p, j, uplo);
                }
                double *out = c + i * ldc + j;
                *out = alpha * sum + (beta == 0.0 ? 0.0 : beta * *out);
            }
        }
        return;
    }
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n;) {
            const size_t vl = __riscv_vsetvl_e64m2(n - j);
            vfloat64m2_t acc = __riscv_vfmv_v_f_f64m2(0.0, vl);
            if (alpha != 0.0) {
                for (size_t p = 0; p < m; ++p) {
                    const double factor = symmetric_at(a, lda, i, p, uplo);
                    const vfloat64m2_t bp = __riscv_vle64_v_f64m2(b + p * ldb + j, vl);
                    acc = __riscv_vfmacc_vf_f64m2(acc, factor, bp, vl);
                }
            }
            double *out = c + i * ldc + j;
            acc = __riscv_vfmul_vf_f64m2(acc, alpha, vl);
            if (beta != 0.0)
                acc = __riscv_vfmacc_vf_f64m2(acc, beta, __riscv_vle64_v_f64m2(out, vl), vl);
            __riscv_vse64_v_f64m2(out, acc, vl);
            j += vl;
        }
    }
}
