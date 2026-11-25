typedef long long BLASLONG;

#include <riscv_vector.h>
double ddot_k(BLASLONG n, double *x, BLASLONG inc_x, double *y, BLASLONG inc_y)
{
    double dot = 0.0;

    if (n <= 0) return dot;

    // 主路径：单位步长向量加速
    if (inc_x == 1 && inc_y == 1) {
        BLASLONG i = 0;
        size_t vlmax = __riscv_vsetvlmax_e64m8();
        vfloat64m8_t vsum = __riscv_vfmv_v_f_f64m8(0.0, vlmax);

        for (; i <= n - vlmax; i += vlmax) {
            vfloat64m8_t vx = __riscv_vle64_v_f64m8(x + i, vlmax);
            vfloat64m8_t vy = __riscv_vle64_v_f64m8(y + i, vlmax);
            vsum = __riscv_vfmacc_vv_f64m8(vsum, vx, vy, vlmax);
        }

        // 水平归约
        vfloat64m1_t vred = __riscv_vfredusum_vs_f64m8_f64m1(vsum, __riscv_vfmv_v_f_f64m1(0.0, 1), vlmax);
        dot = __riscv_vfmv_f_s_f64m1_f64(vred);

        // 尾部标量处理
        for (; i < n; i++) {
            dot += x[i] * y[i];
        }
    }
    // 通用 strided 路径
    else {
        BLASLONG ix = 0, iy = 0;
        for (BLASLONG i = 0; i < n; i++) {
            dot += x[ix] * y[iy];
            ix += inc_x;
            iy += inc_y;
        }
    }

    return dot;
}
