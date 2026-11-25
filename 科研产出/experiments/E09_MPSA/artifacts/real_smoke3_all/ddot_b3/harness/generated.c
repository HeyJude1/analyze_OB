#include <riscv_vector.h>
#include <stddef.h>
typedef long long BLASLONG;
double ddot_k(BLASLONG n, double *x, BLASLONG inc_x, double *y, BLASLONG inc_y)
{
    if (n <= 0) return 0.0;

    double dot = 0.0;
    size_t vlmax = __riscv_vsetvlmax_e64m8();
    vfloat64m8_t vsum = __riscv_vfmv_v_f_f64m8(0.0, vlmax);

    if (inc_x == 1 && inc_y == 1) {
        BLASLONG i = 0;
        for (; i < n; ) {
            size_t vl = __riscv_vsetvl_e64m8(n - i);
            vfloat64m8_t vx = __riscv_vle64_v_f64m8(x + i, vl);
            vfloat64m8_t vy = __riscv_vle64_v_f64m8(y + i, vl);
            vsum = __riscv_vfmacc_vv_f64m8(vsum, vx, vy, vl);
            i += vl;
        }
    } else if (inc_y == 1) {
        BLASLONG i = 0;
        BLASLONG ix = 0;
        for (; i < n; ) {
            size_t vl = __riscv_vsetvl_e64m8(n - i);
            vfloat64m8_t vx = __riscv_vlse64_v_f64m8(x + ix, inc_x * sizeof(double), vl);
            vfloat64m8_t vy = __riscv_vle64_v_f64m8(y + i, vl);
            vsum = __riscv_vfmacc_vv_f64m8(vsum, vx, vy, vl);
            i += vl;
            ix += vl * inc_x;
        }
    } else {
        BLASLONG i = 0;
        BLASLONG ix = 0;
        BLASLONG iy = 0;
        for (; i < n; ) {
            size_t vl = __riscv_vsetvl_e64m8(n - i);
            vfloat64m8_t vx = __riscv_vlse64_v_f64m8(x + ix, inc_x * sizeof(double), vl);
            vfloat64m8_t vy = __riscv_vlse64_v_f64m8(y + iy, inc_y * sizeof(double), vl);
            vsum = __riscv_vfmacc_vv_f64m8(vsum, vx, vy, vl);
            i += vl;
            ix += vl * inc_x;
            iy += vl * inc_y;
        }
    }

    vfloat64m1_t vred = __riscv_vfredusum_vs_f64m8_f64m1(vsum, __riscv_vfmv_v_f_f64m1(0.0, 1), vlmax);
    dot = __riscv_vfmv_f_s_f64m1_f64(vred);

    return dot;
}
