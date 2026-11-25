#include "opbench.h"

void op_dgemm(size_t m, size_t n, size_t k, double alpha, const double *a, size_t lda,
              const double *b, size_t ldb, double beta, double *c, size_t ldc) {
    if (alpha == 0.0 || k == 0) {
        if (beta != 0.0) {
            for (size_t i = 0; i < m; ++i) {
                for (size_t j = 0; j < n; ++j) {
                    c[i * ldc + j] *= beta;
                }
            }
        }
        return;
    }

    const double *pa = a;
    double *pc = c;

    for (size_t j = 0; j < n; j += 4) {
        for (size_t i = 0; i < m; ++i) {
            pc[i] += pa[i * lda];
        }
        pa += 4 * lda;
        pc += 4;
    }

    if (n % 4 != 0) {
        size_t j = n - n % 4;
        for (size_t i = 0; i < m; ++i) {
            pc[i] += pa[i * lda];
        }
    }
}
