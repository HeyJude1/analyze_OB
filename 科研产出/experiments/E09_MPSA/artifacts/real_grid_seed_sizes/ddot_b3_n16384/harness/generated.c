typedef long long BLASLONG;

#include <riscv_vector.h>
double ddot_k(BLASLONG n, double *x, BLASLONG inc_x, double *y, BLASLONG inc_y)
{
    if (n <= 0) return 0.0;

    double dot = 0.0;
    size_t vlmax = __riscv_vsetvlmax_e64m8();
    vfloat64m8_t vsum = __riscv_vfmv_v_f_f64m8(0.0, vlmax);

    if (inc_x == 1 && inc_y == 1) {
        /* 单位步长快速路径：vle64 + vfmvvp */
        BLASLONG i = 0;
        for (; i < n; i += vlmax) {
            size_t vl = __riscv_vsetvl_e64m8(n - i);
            vfloat64m8_t vx = __riscv_vle64_v_f64m8(x + i, vl);
            vfloat64m8_t vy = __riscv_vle64_v_f64m8(y + i, vl);
            vsum = __riscv_vfmvvp_vv_f64m8(vsum, vx, vy, vl);
        }
        /* 归约 */
        vfloat64m1_t vred = __riscv_vfredusum_vs_f64m8_f64m1(vsum, __riscv_vfmv_v_f_f64m1(0.0, 1), vlmax);
        dot = __riscv_vfmv_f_s_f64m1_f64(vred);
        /* 尾部标量处理 */
        for (BLASLONG j = (n / vlmax) * vlmax; j < n; j++) {
            dot += x[j] * y[j];
        }
    } else if (inc_y == 1) {
        /* 跨步 x：vlse64 + 连续 y */
        BLASLONG i = 0;
        for (; i < n; i += vlmax) {
            size_t vl = __riscv_vsetvl_e64m8(n - i);
            vfloat64m8_t vx = __riscv_vlse64_v_f64m8(x + i * inc_x, inc_x * sizeof(double), vl);
            vfloat64m8_t vy = __riscv_vle64_v_f64m8(y + i, vl);
            vsum = __riscv_vfmvvp_vv_f64m8(vsum, vx, vy, vl);
        }
        /* 归约 */
        vfloat64m1_t vred = __riscv_vfredusum_vs_f64m8_f64m1(vsum, __riscv_vfmv_v_f_f64m1(0.0, 1), vlmax);
        dot = __riscv_vfmv_f_s_f64m1_f64(vred);
        /* 尾部标量处理 */
        for (BLASLONG j = (n / vlmax) * vlmax; j < n; j++) {
            dot += x[j * inc_x] * y[j];
        }
    } else {
        /* 通用 strided 路径 */
        BLASLONG i = 0;
        while (i < n) {
            dot += x[i * inc_x] * y[i * inc_y];
            i++;
        }
    }

    return dot;
}
