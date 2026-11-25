typedef long long BLASLONG;

#include <riscv_vector.h>
double ddot_k(BLASLONG n, double *x, BLASLONG inc_x, double *y, BLASLONG inc_y)
{
    double dot = 0.0;
    if (n <= 0) return dot;

    if (inc_x == 1 && inc_y == 1) {
        /* 单位步长快速路径：vle64/vse64 */
        BLASLONG i = 0;
        size_t vl;
        vfloat64m8_t vsum = __riscv_vfmv_v_f_f64m8(0.0, 8);
        while (i < n) {
            vl = __riscv_vsetvl_e64m8(n - i);
            vfloat64m8_t vx = __riscv_vle64_v_f64m8(x + i, vl);
            vfloat64m8_t vy = __riscv_vle64_v_f64m8(y + i, vl);
            vsum = __riscv_vfmacc_vv_f64m8(vsum, vx, vy, vl);
            i += vl;
        }
        dot = __riscv_vfmv_f_s_f64m8_f64(vsum);
    } else if (inc_y == 1) {
        /* 跨步 x：vlse64 + 连续 y */
        BLASLONG i = 0;
        size_t vl;
        vfloat64m8_t vsum = __riscv_vfmv_v_f_f64m8(0.0, 8);
        while (i < n) {
            vl = __riscv_vsetvl_e64m8(n - i);
            vfloat64m8_t vx = __riscv_vlse64_v_f64m8(x, inc_x * sizeof(double), vl);
            vfloat64m8_t vy = __riscv_vle64_v_f64m8(y + i, vl);
            vsum = __riscv_vfmacc_vv_f64m8(vsum, vx, vy, vl);
            x += inc_x * vl;
            y += vl;
            i += vl;
        }
        dot = __riscv_vfmv_f_s_f64m8_f64(vsum);
    } else {
        /* 通用 strided 路径 */
        BLASLONG i = 0;
        size_t vl;
        vfloat64m8_t vsum = __riscv_vfmv_v_f_f64m8(0.0, 8);
        while (i < n) {
            vl = __riscv_vsetvl_e64m8(n - i);
            vfloat64m8_t vx = __riscv_vlse64_v_f64m8(x, inc_x * sizeof(double), vl);
            vfloat64m8_t vy = __riscv_vlse64_v_f64m8(y, inc_y * sizeof(double), vl);
            vsum = __riscv_vfmacc_vv_f64m8(vsum, vx, vy, vl);
            x += inc_x * vl;
            y += inc_y * vl;
            i += vl;
        }
        dot = __riscv_vfmv_f_s_f64m8_f64(vsum);
    }

    return dot;
}
