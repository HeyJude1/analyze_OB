#include "opbench.h"
#include <immintrin.h>

static double stored_dot(const double *row, const double *x, ptrdiff_t incx, size_t start, size_t end) {
    double sum = 0.0;
    size_t j = start;
    if (incx == 1 && end - j >= 4) {
        __m256d acc = _mm256_setzero_pd();
        for (; end - j >= 4; j += 4)
            acc = _mm256_fmadd_pd(_mm256_loadu_pd(row + j), _mm256_loadu_pd(x + j), acc);
        double lanes[4];
        _mm256_storeu_pd(lanes, acc);
        sum = lanes[0] + lanes[1] + lanes[2] + lanes[3];
    }
    for (; j < end; ++j) sum += row[j] * x[(ptrdiff_t)j * incx];
    return sum;
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
