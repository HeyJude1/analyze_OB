typedef long long BLASLONG;

#include <riscv_vector.h>
double ddot_k(BLASLONG n, double *x, BLASLONG inc_x, double *y, BLASLONG inc_y)
{
    if (n <= 0) return 0.0;

    double dot = 0.0;

    if (inc_x == 1 && inc_y == 1) {
        BLASLONG i = 0;
        const BLASLONG vl = __riscv_vsetvlmax_e64m8();
        vfloat64m8_t vsum = __riscv_vmvinf_f64m8();

        for (; i <= n - vl; i += vl) {
            vfloat64m8_t vx = __riscv_vle64_v_f64m8(x + i, vl);
            vfloat64m8_t vy = __riscv_vle64_v_f64m8(y + i, vl);
            vsum = __riscv_vfmacc_vv_f64m8(vsum, vx, vy, vl);
        }

        dot = __riscv_vfmv_f_s_f64m8(vsum, vl);

        for (; i < n; i++) {
            dot += x[i] * y[i];
        }
    } else {
        BLASLONG ix = 0, iy = 0;
        BLASLONG i;

        for (i = 0; i < n; i++) {
            dot += x[ix] * y[iy];
            ix += inc_x;
            iy += inc_y;
        }
    }

    return dot;
}
