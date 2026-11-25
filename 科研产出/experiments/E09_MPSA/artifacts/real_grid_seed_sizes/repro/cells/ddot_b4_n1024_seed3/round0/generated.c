#include <riscv_vector.h>
typedef long long BLASLONG;
double ddot_k(BLASLONG n, double *x, BLASLONG inc_x, double *y, BLASLONG inc_y)
{
    if (n <= 0) return 0.0;

    double dot = 0.0;
    size_t vlmax = __riscv_vsetvlmax_e64m8();
    vfloat64m8_t vsum = __riscv_vfmv_v_f_f64m8(0.0, vlmax);

    if (inc_x == 1 && inc_y == 1) {
        BLASLONG i = 0;
        for (; i <= n - vlmax; i += vlmax) {
            vfloat64m8_t vx = __riscv_vle64_v_f64m8(x + i, vlmax);
            vfloat64m8_t vy = __riscv_vle64_v_f64m8(y + i, vlmax);
            vsum = __riscv_vfmacc_vv_f64m8(vsum, vx, vy, vlmax);
        }

        // Tail handling
        size_t vl = __riscv_vsetvl_e64m8(n - i);
        if (vl > 0) {
            vfloat64m8_t vx = __riscv_vle64_v_f64m8(x + i, vl);
            vfloat64m8_t vy = __riscv_vle64_v_f64m8(y + i, vl);
            vsum = __riscv_vfmacc_vv_f64m8(vsum, vx, vy, vl);
        }
    } else {
        BLASLONG i = 0;
        for (; i <= n - vlmax; i += vlmax) {
            vfloat64m8_t vsum_lane = __riscv_vfmv_v_f_f64m8(0.0, vlmax);
            for (size_t lane = 0; lane < vlmax; ++lane) {
                vsum_lane = __riscv_vfmacc_vf_f64m8(vsum_lane, ((double *)x)[lane * inc_x + i], ((double *)y)[lane * inc_y + i], vlmax);
            }
            vsum = __riscv_vfadd_vv_f64m8(vsum, vsum_lane, vlmax);
        }

        // Tail handling
        size_t vl = __riscv_vsetvl_e64m8(n - i);
        if (vl > 0) {
            vfloat64m8_t vsum_lane = __riscv_vfmv_v_f_f64m8(0.0, vl);
            for (size_t lane = 0; lane < vl; ++lane) {
                vsum_lane = __riscv_vfmacc_vf_f64m8(vsum_lane, ((double *)x)[lane * inc_x + i], ((double *)y)[lane * inc_y + i], vl);
            }
            vsum = __riscv_vfadd_vv_f64m8(vsum, vsum_lane, vl);
        }
    }

    // Horizontal reduction
    vfloat64m1_t vred = __riscv_vfredusum_vs_f64m8_f64m1(vsum, __riscv_vfmv_v_f_f64m1(0.0, 1), vlmax);
    dot = __riscv_vfmv_f_s_f64m1_f64(vred);

    return dot;
}
