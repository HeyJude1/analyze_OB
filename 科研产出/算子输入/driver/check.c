#include "opbench.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define GUARD 719314.125
#define ARRAY_LEN(a) (sizeof(a) / sizeof((a)[0]))

static uint32_t rng = 9147309;
static unsigned failures;
static unsigned cases;

static double random_value(void) {
    rng = 1664525u * rng + 1013904223u;
    return (double)((int)((rng >> 8) % 2001u) - 1000) / 377.0;
}

#if !defined(OPBENCH_TASK_DSCAL) && !defined(OPBENCH_TASK_DSYMV) && !defined(OPBENCH_TASK_DSYMM)
static void fill(double *dst, size_t n) {
    for (size_t i = 0; i < n; ++i) dst[i] = random_value();
}
#endif

static void expect_close(const char *name, size_t i, double observed, long double expected) {
    /* Finite inputs on a moderate scale. Stronger numerical tests are separate. */
    const long double error = fabsl((long double)observed - expected);
    const long double tolerance = 2e-12L + 3e-12L * fabsl(expected);
    if (!isfinite(observed) || error > tolerance) {
        if (failures < 10) {
            fprintf(stderr, "%s case %u cell %zu: got %.17g expected %.17g\n", name, cases, i,
                    observed, (double)expected);
        }
        ++failures;
    }
}

#if defined(OPBENCH_TASK_DGEMV) || defined(OPBENCH_TASK_DGEMM) || defined(OPBENCH_TASK_CONV1D) || \
    defined(OPBENCH_TASK_DSCAL) || defined(OPBENCH_TASK_DSYMV) || defined(OPBENCH_TASK_DSYMM)
static void expect_guard(const char *name, size_t i, double observed) {
    if (observed != GUARD) {
        if (failures < 10) fprintf(stderr, "%s case %u modified guard %zu\n", name, cases, i);
        ++failures;
    }
}
#endif

#if !defined(OPBENCH_TASK_DDOT) && !defined(OPBENCH_TASK_DSCAL)
static void expect_unchanged(const char *name, const double *p, const double *copy, size_t len) {
    if (memcmp(p, copy, len * sizeof(*p)) != 0) {
        if (failures < 10) fprintf(stderr, "%s case %u modified a read-only input\n", name, cases);
        ++failures;
    }
}
#endif

#if defined(OPBENCH_TASK_DAXPY) || defined(OPBENCH_TASK_DDOT) || defined(OPBENCH_TASK_DGEMV) || \
    defined(OPBENCH_TASK_DSCAL) || defined(OPBENCH_TASK_DASUM) || defined(OPBENCH_TASK_DSYMV)
static size_t logical_offset(size_t n, ptrdiff_t stride) {
    return n != 0 && stride < 0 ? (n - 1) * (size_t)(-stride) : 0;
}
#endif

#if defined(OPBENCH_TASK_DAXPY) || defined(OPBENCH_TASK_DDOT)
static void check_vectors(void) {
    const size_t sizes[] = {0, 1, 3, 4, 7, 8, 9, 15, 16, 17, 33, 131};
    const ptrdiff_t strides[][2] = {{1, 1}, {2, 1}, {1, 3}, {-2, 3}, {1, -2}};
    for (size_t a = 0; a < ARRAY_LEN(sizes); ++a) {
        const size_t n = sizes[a];
        for (size_t s = 0; s < ARRAY_LEN(strides); ++s) {
            ++cases;
            double x[400], y[400], x_before[400], y_before[400];
            const ptrdiff_t incx = strides[s][0], incy = strides[s][1];
            const size_t x0 = 1 + logical_offset(n, incx), y0 = 1 + logical_offset(n, incy);
            fill(x, ARRAY_LEN(x));
            fill(y, ARRAY_LEN(y));
            memcpy(x_before, x, sizeof(x));
            memcpy(y_before, y, sizeof(y));
#if defined(OPBENCH_TASK_DAXPY)
            const double alpha = (a % 3 == 0) ? 0.0 : -1.73;
            op_daxpy(n, alpha, alpha == 0.0 ? NULL : x + x0, incx, y + y0, incy);
            long double expected[400];
            for (size_t i = 0; i < ARRAY_LEN(expected); ++i) expected[i] = y_before[i];
            for (size_t i = 0; i < n; ++i) {
                expected[(ptrdiff_t)y0 + (ptrdiff_t)i * incy] +=
                    (long double)alpha * x_before[(ptrdiff_t)x0 + (ptrdiff_t)i * incx];
            }
            for (size_t i = 0; i < ARRAY_LEN(y); ++i) expect_close("daxpy", i, y[i], expected[i]);
            expect_unchanged("daxpy x", x, x_before, ARRAY_LEN(x));
#else
            long double expected = 0.0L;
            for (size_t i = 0; i < n; ++i) {
                expected += (long double)x_before[(ptrdiff_t)x0 + (ptrdiff_t)i * incx] *
                    y_before[(ptrdiff_t)y0 + (ptrdiff_t)i * incy];
            }
            const double got = op_ddot(n, n == 0 ? NULL : x + x0, incx, n == 0 ? NULL : y + y0, incy);
            expect_close("ddot", 0, got, expected);
            if (memcmp(x, x_before, sizeof(x)) != 0 || memcmp(y, y_before, sizeof(y)) != 0) ++failures;
#endif
        }
    }
}
#endif

#if defined(OPBENCH_TASK_DGEMV)
static void check_dgemv(void) {
    const size_t shapes[][2] = {{0, 9}, {5, 0}, {1, 1}, {3, 5}, {5, 8}, {9, 17}, {11, 37}};
    const double scales[][2] = {{1.0, 0.0}, {-1.131, 1.33}, {0.0, -0.6}, {0.0, 0.0}};
    const ptrdiff_t strides[][2] = {{1, 1}, {2, 3}, {-2, -2}};
    for (size_t sh = 0; sh < ARRAY_LEN(shapes); ++sh) {
        const size_t m = shapes[sh][0], n = shapes[sh][1], lda = n + 3;
        for (size_t s = 0; s < ARRAY_LEN(strides); ++s) {
            const ptrdiff_t incx = strides[s][0], incy = strides[s][1];
            for (size_t p = 0; p < ARRAY_LEN(scales); ++p) {
                ++cases;
                const double alpha = scales[p][0], beta = scales[p][1];
                double a[480], x[96], y[48], a_before[480], x_before[96], y_before[48];
                const size_t x0 = 1 + logical_offset(n, incx), y0 = 1 + logical_offset(m, incy);
                fill(a, ARRAY_LEN(a));
                fill(x, ARRAY_LEN(x));
                for (size_t i = 0; i < ARRAY_LEN(y); ++i) y[i] = GUARD;
                for (size_t i = 0; i < m; ++i) y[(ptrdiff_t)y0 + (ptrdiff_t)i * incy] = beta == 0.0 ? NAN : random_value();
                memcpy(a_before, a, sizeof(a));
                memcpy(x_before, x, sizeof(x));
                memcpy(y_before, y, sizeof(y));
                op_dgemv(m, n, alpha, n == 0 || alpha == 0.0 ? NULL : a, lda,
                         n == 0 || alpha == 0.0 ? NULL : x + x0, incx, beta, y + y0, incy);
                unsigned char written[48] = {0};
                for (size_t i = 0; i < m; ++i) {
                    const size_t index = (size_t)((ptrdiff_t)y0 + (ptrdiff_t)i * incy);
                    long double sum = 0.0L;
                    for (size_t j = 0; j < n; ++j) sum += (long double)a_before[i * lda + j] * x_before[(ptrdiff_t)x0 + (ptrdiff_t)j * incx];
                    const long double old = beta == 0.0 ? 0.0L : (long double)beta * y_before[index];
                    expect_close("dgemv", index, y[index], (long double)alpha * sum + old);
                    written[index] = 1;
                }
                for (size_t i = 0; i < ARRAY_LEN(y); ++i) if (!written[i]) expect_guard("dgemv", i, y[i]);
                expect_unchanged("dgemv a", a, a_before, ARRAY_LEN(a));
                expect_unchanged("dgemv x", x, x_before, ARRAY_LEN(x));
            }
        }
    }
}
#endif

#if defined(OPBENCH_TASK_DGEMM)
static void check_dgemm(void) {
    const size_t shapes[][3] = {{0, 9, 3}, {5, 0, 9}, {5, 9, 0}, {1, 1, 1}, {3, 5, 7}, {6, 8, 12}, {9, 17, 13}, {11, 25, 19}};
    const double scales[][2] = {{1.0, 0.0}, {-1.131, 1.33}, {0.0, -0.6}, {0.0, 0.0}};
    for (size_t sh = 0; sh < ARRAY_LEN(shapes); ++sh) {
        const size_t m = shapes[sh][0], n = shapes[sh][1], k = shapes[sh][2];
        const size_t lda = k + 1, ldb = n + 2, ldc = n + 3;
        for (size_t p = 0; p < ARRAY_LEN(scales); ++p) {
            ++cases;
            const double alpha = scales[p][0], beta = scales[p][1];
            double a[256], b[560], c[352], a_before[256], b_before[560], c_before[352];
            fill(a, ARRAY_LEN(a));
            fill(b, ARRAY_LEN(b));
            for (size_t j = 0; j < ARRAY_LEN(c); ++j) c[j] = GUARD;
            for (size_t i = 0; i < m; ++i) {
                for (size_t j = 0; j < n; ++j) c[1 + i * ldc + j] = beta == 0.0 ? NAN : random_value();
            }
            memcpy(a_before, a, sizeof(a));
            memcpy(b_before, b, sizeof(b));
            memcpy(c_before, c, sizeof(c));
            op_dgemm(m, n, k, alpha, k == 0 || alpha == 0.0 ? NULL : a, lda,
                     k == 0 || alpha == 0.0 ? NULL : b, ldb, beta, c + 1, ldc);
            unsigned char written[352] = {0};
            for (size_t i = 0; i < m; ++i) {
                for (size_t j = 0; j < n; ++j) {
                    long double sum = 0.0L;
                    for (size_t d = 0; d < k; ++d) sum += (long double)a_before[i * lda + d] * b_before[d * ldb + j];
                    const size_t index = 1 + i * ldc + j;
                    const long double old = beta == 0.0 ? 0.0L : (long double)beta * c_before[index];
                    expect_close("dgemm", index, c[index], (long double)alpha * sum + old);
                    written[index] = 1;
                }
            }
            for (size_t i = 0; i < ARRAY_LEN(c); ++i) if (!written[i]) expect_guard("dgemm", i, c[i]);
            expect_unchanged("dgemm a", a, a_before, ARRAY_LEN(a));
            expect_unchanged("dgemm b", b, b_before, ARRAY_LEN(b));
        }
    }
}
#endif

#if defined(OPBENCH_TASK_CONV1D)
static void check_conv1d(void) {
    const size_t shapes[][2] = {{0, 1}, {3, 0}, {2, 3}, {1, 1}, {7, 3}, {7, 7}, {35, 7}, {100, 11}};
    for (size_t sh = 0; sh < ARRAY_LEN(shapes); ++sh) {
        ++cases;
        const size_t n = shapes[sh][0], k = shapes[sh][1], len = k != 0 && k <= n ? n - k + 1 : 0;
        double x[102], w[16], y[102], x_before[102], w_before[16];
        fill(x, ARRAY_LEN(x));
        fill(w, ARRAY_LEN(w));
        for (size_t i = 0; i < ARRAY_LEN(y); ++i) y[i] = GUARD;
        memcpy(x_before, x, sizeof(x));
        memcpy(w_before, w, sizeof(w));
        op_conv1d(n, k, len == 0 ? NULL : x + 1, len == 0 ? NULL : w + 1, y + 1);
        expect_guard("conv1d", 0, y[0]);
        for (size_t i = 0; i < len; ++i) {
            long double sum = 0.0L;
            for (size_t j = 0; j < k; ++j) sum += (long double)x_before[i + j + 1] * w_before[k - j];
            expect_close("conv1d", i + 1, y[i + 1], sum);
        }
        for (size_t i = len + 1; i < ARRAY_LEN(y); ++i) expect_guard("conv1d", i, y[i]);
        expect_unchanged("conv1d x", x, x_before, ARRAY_LEN(x));
        expect_unchanged("conv1d w", w, w_before, ARRAY_LEN(w));
    }
}
#endif

#if defined(OPBENCH_TASK_DSCAL) || defined(OPBENCH_TASK_DASUM)
static void check_vector_holdout(void) {
    const size_t sizes[] = {0, 1, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 67, 131};
    const ptrdiff_t strides[] = {1, 2, 3, -1, -2, -3};
#if defined(OPBENCH_TASK_DSCAL)
    const double scales[] = {0.0, 1.0, -1.0, 0.301};
#endif
    for (size_t shape = 0; shape < ARRAY_LEN(sizes); ++shape) {
        const size_t n = sizes[shape];
        for (size_t s = 0; s < ARRAY_LEN(strides); ++s) {
            const ptrdiff_t inc = strides[s];
            const ptrdiff_t first = (ptrdiff_t)(1 + logical_offset(n, inc));
#if defined(OPBENCH_TASK_DSCAL)
            for (size_t p = 0; p < ARRAY_LEN(scales); ++p) {
                ++cases;
                double x[400], before[400];
                unsigned char written[400] = {0};
                for (size_t i = 0; i < ARRAY_LEN(x); ++i) x[i] = GUARD;
                for (size_t i = 0; i < n; ++i) {
                    const ptrdiff_t at = first + (ptrdiff_t)i * inc;
                    x[at] = random_value();
                    written[at] = 1;
                }
                memcpy(before, x, sizeof(x));
                op_dscal(n, scales[p], n == 0 ? NULL : x + first, inc);
                for (size_t i = 0; i < ARRAY_LEN(x); ++i) {
                    if (written[i]) expect_close("dscal", i, x[i], (long double)scales[p] * before[i]);
                    else expect_guard("dscal", i, x[i]);
                }
            }
#else
            for (size_t distribution = 0; distribution < 3; ++distribution) {
                ++cases;
                double x[400], before[400];
                long double expected = 0.0L;
                fill(x, ARRAY_LEN(x));
                for (size_t i = 0; i < n; ++i) {
                    const ptrdiff_t at = first + (ptrdiff_t)i * inc;
                    if (distribution == 0) x[at] = i % 3 == 0 ? -0.0 : -fabs(x[at]);
                    else if (distribution == 1 && i % 5 == 0) x[at] *= 17.0;
                    else if (distribution == 2) x[at] = i % 2 == 0 ? -2.125 : 2.125;
                    expected += fabsl((long double)x[at]);
                }
                memcpy(before, x, sizeof(x));
                const double result = op_dasum(n, n == 0 ? NULL : x + first, inc);
                expect_close("dasum absolute sum", 0, result, expected);
                expect_unchanged("dasum x", x, before, ARRAY_LEN(x));
            }
#endif
        }
    }
}
#endif

#if defined(OPBENCH_TASK_DSYMV) || defined(OPBENCH_TASK_DSYMM)
enum { SYM_LIMIT = 41, SYM_BUFFER_SIZE = 48 * 48 + 4 };

/* Keep a full canonical matrix as the independent reference. The candidate sees
 * only a triangular extract, with the opposite triangle and padding poisoned.
 */
static void make_symmetric(size_t n, size_t ld, char uplo, double *dense, double *stored) {
    for (size_t i = 0; i < SYM_BUFFER_SIZE; ++i) stored[i] = NAN;
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j <= i; ++j) {
            const double entry = random_value() + (i == j ? 3.0 : 0.0);
            dense[i * n + j] = entry;
            dense[j * n + i] = entry;
            if (uplo == 'L') stored[i * ld + j] = entry;
            else stored[j * ld + i] = entry;
        }
    }
}

static const double symmetric_scales[][2] = {
    {1.0, 0.0}, {-1.131, 1.33}, {0.0, -0.6}, {0.0, 0.0}, {0.0, 1.0}, {0.75, 1.0}
};
#endif

#if defined(OPBENCH_TASK_DSYMV)
static void check_dsymv(void) {
    const size_t sizes[] = {0, 1, 2, 3, 4, 5, 7, 8, 9, 17, 37, 41};
    const ptrdiff_t strides[][2] = {{1, 1}, {2, 3}, {-2, 1}, {1, -3}, {-1, -2}};
    const char triangles[] = {'U', 'L'};
    for (size_t sz = 0; sz < ARRAY_LEN(sizes); ++sz) {
        const size_t n = sizes[sz];
        for (size_t u = 0; u < ARRAY_LEN(triangles); ++u) {
            const char uplo = triangles[u];
            for (size_t s = 0; s < ARRAY_LEN(strides); ++s) {
                const ptrdiff_t incx = strides[s][0], incy = strides[s][1];
                const ptrdiff_t x0 = 1 + (ptrdiff_t)logical_offset(n, incx);
                const ptrdiff_t y0 = 1 + (ptrdiff_t)logical_offset(n, incy);
                for (size_t p = 0; p < ARRAY_LEN(symmetric_scales); ++p) {
                    ++cases;
                    const double alpha = symmetric_scales[p][0], beta = symmetric_scales[p][1];
                    const size_t lda = n + ((s + p) % 2 == 0 ? 0 : 5);
                    double a[SYM_BUFFER_SIZE], a_before[SYM_BUFFER_SIZE], dense[SYM_LIMIT * SYM_LIMIT];
                    double x[128], y[128], x_before[128], y_before[128];
                    make_symmetric(n, lda, uplo, dense, a);
                    for (size_t i = 0; i < ARRAY_LEN(x); ++i) x[i] = NAN;
                    for (size_t i = 0; i < ARRAY_LEN(y); ++i) y[i] = GUARD;
                    for (size_t i = 0; i < n; ++i) {
                        x[x0 + (ptrdiff_t)i * incx] = random_value();
                        y[y0 + (ptrdiff_t)i * incy] = beta == 0.0 ? NAN : random_value();
                    }
                    memcpy(a_before, a, sizeof(a));
                    memcpy(x_before, x, sizeof(x));
                    memcpy(y_before, y, sizeof(y));
                    op_dsymv(uplo, n, alpha, !n || !alpha ? NULL : a, lda,
                             !n || !alpha ? NULL : x + x0, incx, beta, n ? y + y0 : NULL, incy);
                    unsigned char written[128] = {0};
                    for (size_t i = 0; i < n; ++i) {
                        const size_t out = (size_t)(y0 + (ptrdiff_t)i * incy);
                        long double sum = 0.0L;
                        for (size_t j = 0; j < n; ++j)
                            sum += (long double)dense[i * n + j] * x_before[x0 + (ptrdiff_t)j * incx];
                        const long double old = beta == 0.0 ? 0.0L : (long double)beta * y_before[out];
                        expect_close("dsymv triangle/stride/alpha/beta", out, y[out], (long double)alpha * sum + old);
                        written[out] = 1;
                    }
                    for (size_t i = 0; i < ARRAY_LEN(y); ++i) if (!written[i]) expect_guard("dsymv", i, y[i]);
                    if (alpha == 0.0 && beta == 1.0) expect_unchanged("dsymv y, alpha=0 beta=1", y, y_before, ARRAY_LEN(y));
                    expect_unchanged("dsymv a", a, a_before, ARRAY_LEN(a));
                    expect_unchanged("dsymv x", x, x_before, ARRAY_LEN(x));
                }
            }
        }
    }
}
#endif

#if defined(OPBENCH_TASK_DSYMM)
static void check_dsymm(void) {
    const size_t shapes[][2] = {
        {0, 9}, {5, 0}, {1, 1}, {1, 17}, {17, 1}, {3, 5}, {4, 8},
        {9, 17}, {17, 9}, {31, 8}, {11, 37}, {7, 41}
    };
    const char sides[] = {'L', 'R'}, triangles[] = {'U', 'L'};
    for (size_t sh = 0; sh < ARRAY_LEN(shapes); ++sh) {
        const size_t m = shapes[sh][0], n = shapes[sh][1];
        for (size_t s = 0; s < ARRAY_LEN(sides); ++s) {
            const char side = sides[s];
            const size_t order = side == 'L' ? m : n;
            for (size_t u = 0; u < ARRAY_LEN(triangles); ++u) {
                const char uplo = triangles[u];
                for (size_t p = 0; p < ARRAY_LEN(symmetric_scales); ++p) {
                    ++cases;
                    const double alpha = symmetric_scales[p][0], beta = symmetric_scales[p][1];
                    const size_t lda = order + ((p + u) % 2 == 0 ? 0 : 3);
                    const size_t ldb = n + ((p + s) % 2 == 0 ? 0 : 2);
                    const size_t ldc = n + ((p + sh) % 2 == 0 ? 0 : 5);
                    double a[SYM_BUFFER_SIZE], a_before[SYM_BUFFER_SIZE], dense[SYM_LIMIT * SYM_LIMIT];
                    double b[SYM_BUFFER_SIZE], b_before[SYM_BUFFER_SIZE];
                    double c[SYM_BUFFER_SIZE], c_before[SYM_BUFFER_SIZE];
                    make_symmetric(order, lda, uplo, dense, a);
                    for (size_t i = 0; i < ARRAY_LEN(b); ++i) b[i] = NAN;
                    for (size_t i = 0; i < ARRAY_LEN(c); ++i) c[i] = GUARD;
                    for (size_t i = 0; i < m; ++i) {
                        for (size_t j = 0; j < n; ++j) {
                            b[i * ldb + j] = random_value();
                            c[1 + i * ldc + j] = beta == 0.0 ? NAN : random_value();
                        }
                    }
                    memcpy(a_before, a, sizeof(a));
                    memcpy(b_before, b, sizeof(b));
                    memcpy(c_before, c, sizeof(c));
                    op_dsymm(side, uplo, m, n, alpha, !m || !n || !alpha ? NULL : a, lda,
                             !m || !n || !alpha ? NULL : b, ldb, beta, m && n ? c + 1 : NULL, ldc);
                    unsigned char written[SYM_BUFFER_SIZE] = {0};
                    for (size_t i = 0; i < m; ++i) {
                        for (size_t j = 0; j < n; ++j) {
                            long double sum = 0.0L;
                            if (side == 'L') {
                                for (size_t k = 0; k < m; ++k)
                                    sum += (long double)dense[i * m + k] * b_before[k * ldb + j];
                            } else {
                                for (size_t k = 0; k < n; ++k)
                                    sum += (long double)b_before[i * ldb + k] * dense[k * n + j];
                            }
                            const size_t out = 1 + i * ldc + j;
                            const long double old = beta == 0.0 ? 0.0L : (long double)beta * c_before[out];
                            expect_close("dsymm side/triangle/alpha/beta", out, c[out], (long double)alpha * sum + old);
                            written[out] = 1;
                        }
                    }
                    for (size_t i = 0; i < ARRAY_LEN(c); ++i) if (!written[i]) expect_guard("dsymm", i, c[i]);
                    if (alpha == 0.0 && beta == 1.0) expect_unchanged("dsymm c, alpha=0 beta=1", c, c_before, ARRAY_LEN(c));
                    expect_unchanged("dsymm a", a, a_before, ARRAY_LEN(a));
                    expect_unchanged("dsymm b", b, b_before, ARRAY_LEN(b));
                }
            }
        }
    }
}
#endif

int main(void) {
#if defined(OPBENCH_TASK_DAXPY) || defined(OPBENCH_TASK_DDOT)
    check_vectors();
#elif defined(OPBENCH_TASK_DGEMV)
    check_dgemv();
#elif defined(OPBENCH_TASK_DGEMM)
    check_dgemm();
#elif defined(OPBENCH_TASK_CONV1D)
    check_conv1d();
#elif defined(OPBENCH_TASK_DSCAL) || defined(OPBENCH_TASK_DASUM)
    check_vector_holdout();
#elif defined(OPBENCH_TASK_DSYMV)
    check_dsymv();
#elif defined(OPBENCH_TASK_DSYMM)
    check_dsymm();
#else
#error Select an OPBENCH_TASK_* macro
#endif
    printf("%u cases, %u failures\n", cases, failures);
    return failures != 0;
}
