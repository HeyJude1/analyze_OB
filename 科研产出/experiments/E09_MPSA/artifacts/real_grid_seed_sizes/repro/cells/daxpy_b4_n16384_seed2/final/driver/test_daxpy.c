/* daxpy 测试驱动（E09 版，基于 OpOptBench tasks/daxpy/driver/test_daxpy.c v0.1 修改）
 *
 * 与原始 driver 的差异（仅容差与打印，语义/参考实现/调用方式不变）：
 *   1) 增加 reldiff=：LLM 生成代码常用 RVV FMA（vfmacc）融合乘加，逐元素结果与
 *      "先乘后加"的参考实现会在末位舍入上差 ~1 ulp；绝对阈值 1e-12 会把这类
 *      正确实现误判为错误，故判定改为相对误差 reldiff < 1e-12；
 *   2) 打印 reldiff，便于 harness 记录（maxdiff 仍是绝对差）。
 *
 * 编译（WSL 交叉）：riscv64-linux-gnu-gcc -O3 -march=rv64gcv_zvl256b_zba_zbb_zbc_zbs \
 *       -mabi=lp64d -static driver/test_daxpy.c generated.c -lm -o bin
 * 运行（QEMU）：qemu-riscv64 -cpu rv64,v=true,vlen=256,zba=true,zbb=true,zbc=true,zbs=true ./bin 16384 10
 * 退出码：0 = 正确；2 = 数值错误（harness 据此判 correct）
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
    int reps = argc > 2 ? atoi(argv[2]) : 10;
    if (n < 1) { printf("bad n=%ld\n", n); return 1; }
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
    /* 参考实现（与 generic.c 同序：y[i] += da*x[i]） */
    for (long i = 0; i < n; i++) yref[i] += da * x[i];

    /* 正确性（一次性，不参与计时） */
    daxpy_k(n, 0, 0, da, x, 1, y, 1, NULL, 0);
    double maxdiff = 0.0, ymax = 0.0;
    for (long i = 0; i < n; i++) {
        double d = fabs(y[i] - yref[i]);
        if (d > maxdiff) maxdiff = d;
        if (fabs(yref[i]) > ymax) ymax = fabs(yref[i]);
    }
    double reldiff = maxdiff / (ymax > 0.0 ? ymax : 1.0);

    /* 计时（QEMU 数字仅作"能跑完"参考，不是性能结论） */
    double best = 1e30;
    for (int r = 0; r < reps; r++) {
        for (long i = 0; i < n; i++) y[i] = 0.5 * i;
        double t0 = now_ms();
        daxpy_k(n, 0, 0, da, x, 1, y, 1, NULL, 0);
        double dt = now_ms() - t0;
        if (dt < best) best = dt;
    }
    double gbps = best > 0.0 ? (3.0 * n * sizeof(double)) / (best * 1e6) : 0.0;
    printf("daxpy n=%ld maxdiff=%.3e reldiff=%.3e best=%.6f ms (%.2f GB/s)\n",
           n, maxdiff, reldiff, best, gbps);
    return reldiff < 1e-12 ? 0 : 2;
}
