#include <riscv_vector.h>
typedef long long BLASLONG;
double ddot_k(BLASLONG n, double *x, BLASLONG inc_x, double *y, BLASLONG inc_y)
{
    if (n <= 0) return 0.0;

    double dot = 0.0;
    size_t vlmax = __riscv_vsetvlmax_e64m8();

    if (inc_x == 1 && inc_y == 1) {
        /* 单位步长快速路径：vle64 + vdot */
        BLASLONG i = 0;
        vfloat64m8_t vsum = __riscv_vfmv_v_f_f64m8(0.0, vlmax);

        for (; i <= n - vlmax; i += vlmax) {
            size_t vl = __riscv_vsetvl_e64m8(vlmax);
            vfloat64m8_t vx = __riscv_vle64_v_f64m8(x + i, vl);
            vfloat64m8_t vy = __riscv_vle64_v_f64m8(y + i, vl);
            vsum = __riscv_vfmacc_vv_f64m8(vsum, vx, vy, vl);
        }

        /* 水平归约 */
        vfloat64m1_t vred = __riscv_vfredusum_vs_f64m8_f64(vsum, __riscv_vfmv_v_f_f64m1(0.0, 1), vlmax);
        dot = __riscv_vfmv_f_s_f64m1_f64(vred);

        /* 标量尾部处理 */
        for (; i < n; i++) {
            dot += x[i] * y[i];
        }
    } else {
        /* 通用 strided 路径 */
        BLASLONG ix = 0, iy = 0;
        for (BLASLONG i = 0; i < n; i++) {
            dot += x[ix] * y[iy];
            ix += inc_x;
            iy += inc_y;
        }
    }

    return dot;
}
