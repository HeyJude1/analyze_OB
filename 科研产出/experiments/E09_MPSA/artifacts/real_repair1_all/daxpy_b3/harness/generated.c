#include <riscv_vector.h>
typedef long long BLASLONG;
int daxpy_k(BLASLONG n, BLASLONG dummy0, BLASLONG dummy1, double da, double *x, BLASLONG inc_x, double *y, BLASLONG inc_y, double *dummy, BLASLONG dummy2)
{
    if (n <= 0 || da == 0.0) return 0;

    if (inc_x == 1 && inc_y == 1) {
        BLASLONG i = 0;
        size_t vl;

        while (i < n) {
            vl = __riscv_vsetvl_e64m8(n - i);

            vfloat64m8_t vx = __riscv_vle64_v_f64m8(x + i, vl);
            vfloat64m8_t vy = __riscv_vle64_v_f64m8(y + i, vl);
            vfloat64m8_t vres = __riscv_vfmacc_vf_f64m8(vy, da, vx, vl);
            __riscv_vse64_v_f64m8(y + i, vres, vl);

            i += vl;
        }
    } else if (inc_y == 1) {
        BLASLONG i = 0;
        BLASLONG ix = 0;
        size_t vl;

        while (i < n) {
            vl = __riscv_vsetvl_e64m8(n - i);

            vfloat64m8_t vx = __riscv_vlse64_v_f64m8(x + ix, inc_x, vl);
            vfloat64m8_t vy = __riscv_vle64_v_f64m8(y + i, vl);
            vfloat64m8_t vres = __riscv_vfmacc_vf_f64m8(vy, da, vx, vl);
            __riscv_vse64_v_f64m8(y + i, vres, vl);

            i += vl;
            ix += vl * inc_x;
        }
    } else {
        BLASLONG i = 0;
        BLASLONG ix = 0;
        BLASLONG iy = 0;
        size_t vl;

        while (i < n) {
            vl = __riscv_vsetvl_e64m8(n - i);

            vfloat64m8_t vx = __riscv_vlse64_v_f64m8(x + ix, inc_x, vl);
            vfloat64m8_t vy = __riscv_vlse64_v_f64m8(y + iy, inc_y, vl);
            vfloat64m8_t vres = __riscv_vfmacc_vf_f64m8(vy, da, vx, vl);
            __riscv_vse64_v_f64m8(y + iy, vres, vl);

            i += vl;
            ix += vl * inc_x;
            iy += vl * inc_y;
        }
    }

    return 0;
}
