#include <riscv_vector.h>
typedef long long BLASLONG;
int daxpy_k(BLASLONG n, BLASLONG dummy0, BLASLONG dummy1, double da, double *x, BLASLONG inc_x, double *y, BLASLONG inc_y, double *dummy, BLASLONG dummy2)
{
    if (n <= 0) return 0;
    if (da == 0.0) return 0;

    if (inc_x == 1 && inc_y == 1) {
        BLASLONG i = 0;
        const double alpha = da;
        for (; i < n; ) {
            size_t vl = __riscv_vsetvl_e64m8(n - i);
            vfloat64m8_t vx = __riscv_vle64_v_f64m8(x + i, vl);
            vfloat64m8_t vy = __riscv_vle64_v_f64m8(y + i, vl);
            vy = __riscv_vfmacc_vf_f64m8(vy, alpha, vx, vl);
            __riscv_vse64_v_f64m8(y + i, vy, vl);
            i += vl;
        }
    } else if (inc_y == 1) {
        BLASLONG i = 0;
        const double alpha = da;
        for (; i < n; ) {
            size_t vl = __riscv_vsetvl_e64m8(n - i);
            vfloat64m8_t vx = __riscv_vlse64_v_f64m8(x, inc_x, vl);
            vfloat64m8_t vy = __riscv_vle64_v_f64m8(y + i, vl);
            vy = __riscv_vfmacc_vf_f64m8(vy, alpha, vx, vl);
            __riscv_vse64_v_f64m8(y + i, vy, vl);
            i += vl;
            x += vl * inc_x;
        }
    } else {
        BLASLONG i = 0;
        const double alpha = da;
        for (; i < n; ) {
            size_t vl = __riscv_vsetvl_e64m8(n - i);
            vfloat64m8_t vx = __riscv_vlse64_v_f64m8(x, inc_x, vl);
            vfloat64m8_t vy = __riscv_vlse64_v_f64m8(y, inc_y, vl);
            vy = __riscv_vfmacc_vf_f64m8(vy, alpha, vx, vl);
            __riscv_vsse64_v_f64m8(y, inc_y, vy, vl);
            i += vl;
            x += vl * inc_x;
            y += vl * inc_y;
        }
    }

    return 0;
}
