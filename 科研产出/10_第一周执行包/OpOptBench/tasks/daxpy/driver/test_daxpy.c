/* daxpy 测试驱动（v0.1）：验证 generic.c 能否独立编译、跑对、计时。
 * 编译：gcc -O2 -std=gnu99 driver/test_daxpy.c input/generic.c -lm -o daxpy_test
 * 运行：./daxpy_test 16384
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

typedef long long BLASLONG;
extern int daxpy_k(BLASLONG n, BLASLONG d0, BLASLONG d1, double da,
                   double *x, BLASLONG inc_x, double *y, BLASLONG inc_y,
                   double *dummy, BLASLONG d2);

static double now_ms(void) {
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

int main(int argc, char **argv) {
    long n = argc > 1 ? atol(argv[1]) : 16384;
    int reps = argc > 2 ? atoi(argv[2]) : 100;
    double da = 2.5;
    double *x = (double *)malloc(sizeof(double) * n);
    double *y = (double *)malloc(sizeof(double) * n);
    double *yref = (double *)malloc(sizeof(double) * n);
    if (!x || !y || !yref) { printf("OOM\n"); return 1; }
    for (long i = 0; i < n; i++) {
        x[i] = 1.0 / (i + 1);
        y[i] = 0.5 * i;
        yref[i] = y[i];
    }
    /* 参考实现 */
    for (long i = 0; i < n; i++) yref[i] += da * x[i];

    /* 正确性 */
    daxpy_k(n, 0, 0, da, x, 1, y, 1, NULL, 0);
    double maxdiff = 0.0;
    for (long i = 0; i < n; i++) {
        double d = fabs(y[i] - yref[i]);
        if (d > maxdiff) maxdiff = d;
    }
    /* 计时 */
    double best = 1e30;
    for (int r = 0; r < reps; r++) {
        for (long i = 0; i < n; i++) y[i] = 0.5 * i;
        double t0 = now_ms();
        daxpy_k(n, 0, 0, da, x, 1, y, 1, NULL, 0);
        double dt = now_ms() - t0;
        if (dt < best) best = dt;
    }
    printf("daxpy n=%ld maxdiff=%.3e best=%.6f ms  (%.2f GB/s)\n",
           n, maxdiff, best, (3.0 * n * sizeof(double)) / (best * 1e6));
    return maxdiff < 1e-12 ? 0 : 2;
}
