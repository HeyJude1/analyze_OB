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

    size_t bm = m, bn = n, bk = k;
    size_t reg_block_n = 4; // Fixed to 4 accumulators for simplicity

    while (bm > 0 && bn > 0 && bk > 0) {
        size_t i = 0, j = 0, k = 0;

        double acc[reg_block_m][reg_block_n] = {0.0};

        // Main block
        for (; i + reg_block_m <= bm; i += reg_block_m) {
            for (; j + reg_block_n <= bn; j += reg_block_n) {
                for (k = 0; k < bk; ++k) {
                    for (size_t ii = 0; ii < reg_block_m; ++ii) {
                        for (size_t jj = 0; jj < reg_block_n; ++jj) {
                            acc[ii][jj] += pa[i * lda + k] * b[k * ldb + j + jj];
                        }
                    }
                }
            }
        }

        // Tail block
        for (; i < bm; ++i) {
            for (j = 0; j < bn; ++j) {
                double sum = 0.0;
                for (k = 0; k < bk; ++k) {
                    sum += pa[i * lda + k] * b[k * ldb + j];
                }
                pc[i * ldc + j] += alpha * sum;
            }
        }

        pa += reg_block_m * lda;
        pc += reg_block_m;
        bm -= reg_block_m;
        bn -= reg_block_n;
    }
}
