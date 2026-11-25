#include "opbench.h"

void op_dsymv(char uplo, size_t n, double alpha, const double *a, size_t lda,
              const double *x, ptrdiff_t incx, double beta, double *y, ptrdiff_t incy) {
    if (n == 0 || (alpha == 0.0 && beta == 1.0)) return;

    // Vectorization parameters
    const int K_UNROLL = 4;
    const int REG_BLOCK_M = 2;
    const int REG_BLOCK_N = 2;

    // SIMD vector type and operations
    typedef double vdouble __attribute__((vector_size(32)));
    vdouble va, vx;
    double sum[K_UNROLL];

    for (size_t i = 0; i < n; i += K_UNROLL) {
        for (int k = 0; k < K_UNROLL; ++k) {
            sum[k] = 0.0;
        }

        if (alpha != 0.0) {
            for (size_t j = 0; j < n; j += REG_BLOCK_M * REG_BLOCK_N) {
                for (int m = 0; m < REG_BLOCK_M; ++m) {
                    for (int n = 0; n < REG_BLOCK_N; ++n) {
                        const int stored = uplo == 'U' ? i + m <= j + n : i + m >= j + n;
                        if (stored) {
                            va = ((vdouble *)a)[(i + m) * lda / sizeof(double) + j + n];
                            vx = ((vdouble *)x)[n * incx / sizeof(double)];
                            sum[m] += __builtin_reduce_add(va * vx);
                        }
                    }
                }
            }
        }

        for (int k = 0; k < K_UNROLL; ++k) {
            const ptrdiff_t index = (ptrdiff_t)(i + k) * incy;
            y[index] = alpha * sum[k] + (beta == 0.0 ? 0.0 : beta * y[index]);
        }
    }
}
