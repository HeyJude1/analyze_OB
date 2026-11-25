/* Linux paired timing driver. Compile generic and candidate in separate
 * translation units with their op_<task> symbols renamed to base_op/test_op.
 * This driver assumes both implementations passed the task's correctness harness.
 */
#define _POSIX_C_SOURCE 200809L
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <malloc.h>
#endif

#if defined(OPBENCH_TASK_DAXPY)
#define ENTRY op_daxpy
#elif defined(OPBENCH_TASK_DDOT)
#define ENTRY op_ddot
#elif defined(OPBENCH_TASK_DGEMV)
#define ENTRY op_dgemv
#elif defined(OPBENCH_TASK_DGEMM)
#define ENTRY op_dgemm
#elif defined(OPBENCH_TASK_CONV1D)
#define ENTRY op_conv1d
#else
#error Define OPBENCH_TASK_<OP>
#endif

#include "opbench.h"
/* GCC and Clang preserve the canonical ABI under both internal link names. */
extern __typeof__(ENTRY) base_op, test_op;

static volatile double sink;

struct task {
    size_t m, n, k, output_count;
    double *a, *b, *c;
};

static size_t positive_arg(const char *text, size_t limit) {
    for (const char *p = text; *p; ++p) {
        if (*p < '0' || *p > '9') return 0;
    }
    const unsigned long long value = strtoull(text, NULL, 10);
    return value > 0 && value <= limit ? (size_t)value : 0;
}

static double *alloc_array(size_t count, unsigned salt) {
#ifdef _WIN32
    double *buffer = _aligned_malloc(count * sizeof(double), 64);
    if (!buffer) {
#else
    double *buffer = NULL;
    if (posix_memalign((void **)&buffer, 64, count * sizeof(double)) != 0) {
#endif
        fputs("allocation failed\n", stderr);
        exit(2);
    }
    for (size_t i = 0; i < count; ++i) {
        uint32_t x = (uint32_t)i + salt;
        x ^= x >> 16;
        x *= UINT32_C(0x7feb352d);
        x ^= x >> 15;
        buffer[i] = ((int)(x & 1023u) - 512) / 512.0;
    }
    return buffer;
}

static uint64_t now_ns(void) {
#ifdef _WIN32
    LARGE_INTEGER counter, frequency;
    if (!QueryPerformanceCounter(&counter) || !QueryPerformanceFrequency(&frequency)) {
        fputs("QueryPerformanceCounter failed\n", stderr);
        exit(2);
    }
    const uint64_t ticks = (uint64_t)counter.QuadPart;
    const uint64_t hz = (uint64_t)frequency.QuadPart;
    return (ticks / hz) * UINT64_C(1000000000) + (ticks % hz) * UINT64_C(1000000000) / hz;
#else
    struct timespec t;
    if (clock_gettime(CLOCK_MONOTONIC, &t) != 0) {
        perror("clock_gettime");
        exit(2);
    }
    return (uint64_t)t.tv_sec * UINT64_C(1000000000) + (uint64_t)t.tv_nsec;
#endif
}

static uint64_t sample(struct task *t, int candidate, size_t repeats) {
    memset(t->c, 0, t->output_count * sizeof(double));
    double total = 0.0;
    __typeof__(ENTRY) *call = candidate ? test_op : base_op;
    const uint64_t start = now_ns();
    for (size_t i = 0; i < repeats; ++i) {
#if defined(OPBENCH_TASK_DAXPY)
        call(t->n, 0.125, t->a, 1, t->c, 1);
#elif defined(OPBENCH_TASK_DDOT)
        total += call(t->n, t->a, 1, t->b, 1);
#elif defined(OPBENCH_TASK_DGEMV)
        call(t->m, t->n, 0.75, t->a, t->n, t->b, 1, 0.0, t->c, 1);
#elif defined(OPBENCH_TASK_DGEMM)
        call(t->m, t->n, t->k, 0.75, t->a, t->k, t->b, t->n, 0.0, t->c, t->n);
#else
        call(t->n, t->k, t->a, t->b, t->c);
#endif
    }
    const uint64_t elapsed = now_ns() - start;
    sink = total + t->c[t->output_count - 1];
    return elapsed;
}

int main(int argc, char **argv) {
    if (argc != 6) {
        fputs("Usage: measure M N K REPEATS PAIRS (use 1 for unused dimensions)\n", stderr);
        return 2;
    }
    struct task t = {0};
#if defined(OPBENCH_TASK_DGEMM) || defined(OPBENCH_TASK_DGEMV)
    const size_t n_limit = 4096;
#else
    const size_t n_limit = 16777216;
#endif
    t.m = positive_arg(argv[1], 4096);
    t.n = positive_arg(argv[2], n_limit);
    t.k = positive_arg(argv[3], 4096);
    const size_t repeats = positive_arg(argv[4], 10000000);
    const size_t pairs = positive_arg(argv[5], 10000);
    if (!t.m || !t.n || !t.k || !repeats || !pairs) {
        fputs("invalid dimensions, repeat count, or pair count\n", stderr);
        return 2;
    }
    size_t a_count = t.n, b_count = 1;
    t.output_count = t.n;
#if defined(OPBENCH_TASK_DDOT)
    b_count = t.n;
    t.output_count = 1;
#elif defined(OPBENCH_TASK_DGEMV)
    a_count = t.m * t.n;
    b_count = t.n;
    t.output_count = t.m;
#elif defined(OPBENCH_TASK_DGEMM)
    a_count = t.m * t.k;
    b_count = t.k * t.n;
    t.output_count = t.m * t.n;
#elif defined(OPBENCH_TASK_CONV1D)
    if (t.k > t.n) {
        fputs("valid convolution requires K <= N\n", stderr);
        return 2;
    }
    b_count = t.k;
    t.output_count = t.n - t.k + 1;
#endif
    t.a = alloc_array(a_count, 1);
    t.b = alloc_array(b_count, 17);
    t.c = alloc_array(t.output_count, 29);
    sample(&t, 0, 1);
    sample(&t, 1, 1);
    puts("pair,execution_order,m,n,k,repeats,base_total_ns,candidate_total_ns");
    for (size_t i = 0; i < pairs; ++i) {
        uint64_t base, test;
        if (i % 2 == 0) {
            base = sample(&t, 0, repeats);
            test = sample(&t, 1, repeats);
        } else {
            test = sample(&t, 1, repeats);
            base = sample(&t, 0, repeats);
        }
        printf("%zu,%s,%zu,%zu,%zu,%zu,%llu,%llu\n", i,
               i % 2 == 0 ? "base_first" : "candidate_first", t.m, t.n, t.k, repeats,
               (unsigned long long)base, (unsigned long long)test);
    }
#ifdef _WIN32
    _aligned_free(t.a);
    _aligned_free(t.b);
    _aligned_free(t.c);
#else
    free(t.a);
    free(t.b);
    free(t.c);
#endif
    return 0;
}
