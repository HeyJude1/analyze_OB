/* ddot 测试驱动（E09 新增；OpOptBench tasks/ddot 原本没有 driver）
 *
 * 语义：ddot_k(n, x, inc_x, y, inc_y) = Σ_{i=0}^{n-1} x[i*inc_x] * y[i*inc_y]
 *   - generic.c 中的 ddot_k 直接返回 double（无输出指针）；
 *   - n < 0 时返回 0.0。
 * 正确性：参考值用 long double 累加后舍入到 double（"更准的参考"），
 *   判定用相对误差 reldiff < 1e-9（不同求和顺序/向量归约会改变末位舍入）。
 * 计时：best= 为 QEMU 上的最小值，仅作"能跑完"参考，不是性能结论。
 *
 * 编译（WSL 交叉）：riscv64-linux-gnu-gcc -O3 -march=rv64gcv_zvl256b_zba_zbb_zbc_zbs \
 *       -mabi=lp64d -static driver/test_ddot.c generated.c -lm -o bin
 * 运行（QEMU）：qemu-riscv64 -cpu rv64,v=true,vlen=256,zba=true,zbb=true,zbc=true,zbs=true ./bin 16384 10
 * 退出码：0 = 正确；2 = 数值错误（harness 据此判 correct）
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

typedef long long BLASLONG;
extern double ddot_k(BLASLONG n, double *x, BLASLONG inc_x, double *y, BLASLONG inc_y);

static double now_ms(void) {
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

int main(int argc, char **argv) {
    long n = argc > 1 ? atol(argv[1]) : 16384;
    int reps = argc > 2 ? atoi(argv[2]) : 10;
    if (n < 1) { printf("bad n=%ld\n", n); return 1; }
    double *x = (double *)malloc(sizeof(double) * n);
    double *y = (double *)malloc(sizeof(double) * n);
    if (!x || !y) { printf("OOM\n"); return 1; }
    for (long i = 0; i < n; i++) {
        x[i] = 1.0 / (i + 1);
        y[i] = 0.5 * i + 0.1;
    }
    /* 参考值：long double 累加，减少参考本身的舍入误差 */
    long double acc = 0.0L;
    for (long i = 0; i < n; i++) acc += (long double)x[i] * (long double)y[i];
    double ref = (double)acc;

    /* 正确性（一次性，不参与计时） */
    double got = ddot_k(n, x, 1, y, 1);
    double maxdiff = fabs(got - ref);
    double reldiff = maxdiff / (fabs(ref) > 0.0 ? fabs(ref) : 1.0);

    /* 计时（QEMU 数字仅作"能跑完"参考，不是性能结论） */
    double best = 1e30;
    volatile double sink = 0.0;
    for (int r = 0; r < reps; r++) {
        double t0 = now_ms();
        double v = ddot_k(n, x, 1, y, 1);
        double dt = now_ms() - t0;
        sink += v; /* 防优化删循环 */
        if (dt < best) best = dt;
    }
    double gbps = best > 0.0 ? (2.0 * n * sizeof(double)) / (best * 1e6) : 0.0;
    printf("ddot n=%ld maxdiff=%.3e reldiff=%.3e best=%.6f ms (%.2f GB/s) sink=%.3e\n",
           n, maxdiff, reldiff, best, gbps, (double)sink);
    return reldiff < 1e-9 ? 0 : 2;
}
