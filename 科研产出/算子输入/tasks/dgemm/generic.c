#include "opbench.h"

void op_dgemm(size_t m, size_t n, size_t k, double alpha, const double *a, size_t lda,
              const double *b, size_t ldb, double beta, double *c, size_t ldc) {
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            if (alpha != 0.0) {
                for (size_t p = 0; p < k; ++p) sum += a[i * lda + p] * b[p * ldb + j];
            }
            const size_t out = i * ldc + j;
            const double product = k == 0 ? 0.0 : alpha * sum;
            c[out] = product + (beta == 0.0 ? 0.0 : beta * c[out]);
        }
    }
}
