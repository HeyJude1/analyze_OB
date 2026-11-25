#include "opbench.h"

static double symmetric_at(const double *a, size_t lda, size_t i, size_t j, char uplo) {
    const int stored = uplo == 'U' ? i <= j : i >= j;
    return stored ? a[i * lda + j] : a[j * lda + i];
}

void op_dsymm(char side, char uplo, size_t m, size_t n, double alpha,
              const double *a, size_t lda, const double *b, size_t ldb,
              double beta, double *c, size_t ldc) {
    if (m == 0 || n == 0 || (alpha == 0.0 && beta == 1.0)) return;

    // Unroll factor for loop
    const int unroll = 4;
    const size_t m_unrolled = (m + unroll - 1) / unroll * unroll;
    const size_t n_unrolled = (n + unroll - 1) / unroll * unroll;

    // Vectorization parameters
    const int vlen_bits = 256;
    const int elements_per_register = vlen_bits / sizeof(double);
    const int vector_registers = 32;
    const int lmul_max = 8;

    for (size_t i = 0; i < m_unrolled; i += unroll) {
        for (size_t j = 0; j < n_unrolled; j += unroll) {
            double sum[unroll * unroll] = {0.0};

            if (alpha != 0.0 && side == 'L') {
                for (size_t p = 0; p < m_unrolled; ++p) {
                    const double a_val = symmetric_at(a, lda, i, p, uplo);
                    for (int k = 0; k < unroll; ++k) {
                        sum[k * unroll + j] += a_val * b[p * ldb + j + k];
                    }
                }
            } else if (alpha != 0.0) {
                for (size_t p = 0; p < n_unrolled; ++p) {
                    const double b_val = b[i * ldb + p];
                    for (int k = 0; k < unroll; ++k) {
                        sum[k * unroll + j] += b_val * symmetric_at(a, lda, p, j + k, uplo);
                    }
                }
            }

            const size_t index = i * ldc + j;
            for (int k = 0; k < unroll; ++k) {
                c[index + k] = alpha * sum[k * unroll + j] + (beta == 0.0 ? 0.0 : beta * c[index + k]);
            }
        }
    }

    // Handle tail
    for (size_t i = m_unrolled; i < m; ++i) {
        for (size_t j = n_unrolled; j < n; ++j) {
            double sum = 0.0;

            if (alpha != 0.0 && side == 'L') {
                for (size_t p = 0; p < m; ++p)
                    sum += symmetric_at(a, lda, i, p, uplo) * b[p * ldb + j];
            } else if (alpha != 0.0) {
                for (size_t p = 0; p < n; ++p)
                    sum += b[i * ldb + p] * symmetric_at(a, lda, p, j, uplo);
            }

            const size_t index = i * ldc + j;
            c[index] = alpha * sum + (beta == 0.0 ? 0.0 : beta * c[index]);
        }
    }
}
