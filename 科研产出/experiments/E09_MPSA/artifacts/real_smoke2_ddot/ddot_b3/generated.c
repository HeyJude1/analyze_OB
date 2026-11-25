#include <riscv_vector.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
typedef long long BLASLONG;
double ddot_k(BLASLONG n, double *x, BLASLONG inc_x, double *y, BLASLONG inc_y)
{
    BLASLONG i = 0;
    double dot = 0.0;

    if (n <= 0) return dot;

    size_t vlmax = __riscv_vsetvlmax_e64m8();
    vfloat64m8_t vsum = __riscv_vfmv_v_f_f64m8(0.0, vlmax);

    if (inc_x == 1 && inc_y == 1) {
        /* 单位步长快速路径：vle64 + vfmul + vfredusum */
        for (; i <= n - vlmax; i += vlmax) {
            size_t vl = __riscv_vsetvl_e64m8(vlmax);
            vfloat64m8_t vx = __riscv_vle64_v_f64m8(x + i, vl);
            vfloat64m8_t vy = __riscv_vle64_v_f64m8(y + i, vl);
            vsum = __riscv_vfmacc_vv_f64m8(vsum, vx, vy, vl);
        }
        /* 尾部处理 */
        for (; i < n; i++) {
            dot += x[i] * y[i];
        }
    } else if (inc_y == 1) {
        /* 跨步 x：vlse64 + 连续 y */
        for (; i <= n - vlmax; i += vlmax) {
            size_t vl = __riscv_vsetvl_e64m8(vlmax);
            vfloat64m8_t vx = __riscv_vlse64_v_f64m8(x, inc_x * sizeof(double), vl);
            vfloat64m8_t vy = __riscv_vle64_v_f64m8(y + i, vl);
            vsum = __riscv_vfmacc_vv_f64m8(vsum, vx, vy, vl);
            x += inc_x * vlmax;
            y += vlmax;
        }
        /* 尾部处理 */
        for (; i < n; i++) {
            dot += x[0] * y[i];
            x += inc_x;
        }
    } else if (inc_x == 1) {
        /* 跨步 y：vlse64 + 连续 x */
        for (; i <= n - vlmax; i += vlmax) {
            size_t vl = __riscv_vsetvl_e64m8(vlmax);
            vfloat64m8_t vx = __riscv_vle64_v_f64m8(x + i, vl);
            vfloat64m8_t vy = __riscv_vlse64_v_f64m8(y, inc_y * sizeof(double), vl);
            vsum = __riscv_vfmacc_vv_f64m8(vsum, vx, vy, vl);
            y += inc_y * vlmax;
        }
        /* 尾部处理 */
        for (; i < n; i++) {
            dot += x[i] * y[0];
            y += inc_y;
        }
    } else {
        /* 通用 strided 路径 */
        for (; i <= n - vlmax; i += vlmax) {
            size_t vl = __riscv_vsetvl_e64m8(vlmax);
            vfloat64m8_t vx = __riscv_vlse64_v_f64m8(x, inc_x * sizeof(double), vl);
            vfloat64m8_t vy = __riscv_vlse64_v_f64m8(y, inc_y * sizeof(double), vl);
            vsum = __riscv_vfmacc_vv_f64m8(vsum, vx, vy, vl);
            x += inc_x * vlmax;
            y += inc_y * vlmax;
        }
        /* 尾部处理 */
        for (; i < n; i++) {
            dot += x[0] * y[0];
            x += inc_x;
            y += inc_y;
        }
    }

    /* 水平归约 */
    vfloat64m1_t vred = __riscv_vfredusum_vs_f64m8_f64m1(vsum, __riscv_vfmv_v_f_f64m1(0.0, 1), vlmax);
    dot += __riscv_vfmv_f_s_f64m1_f64(vred);

    return dot;
}
