#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <sys/time.h>
#include <riscv_vector.h>
#include"common.h"
# 29 "zzz_ddot_k_rvv_template.c" 2




double ddot_k_rvv(BLASLONG n, double *x, BLASLONG inc_x, double *y, BLASLONG inc_y)

{
    double dot = 0.0;

    if ( n <= 0 ) return(dot);

    size_t vlmax = __riscv_vsetvlmax_e64m8();
    vfloat64m8_t vr = __riscv_vfmv_v_f_f64m8(0, vlmax);

    if(inc_x == 1 && inc_y == 1) {

        for (size_t vl; n > 0; n -= vl, x += vl, y += vl) {
            vl = __riscv_vsetvl_e64m8(n);







            vfloat64m8_t vx = __riscv_vle64_v_f64m8(x, vl);
            vfloat64m8_t vy = __riscv_vle64_v_f64m8(y, vl);

            vr = __riscv_vfmacc_vv_f64m8_tu(vr, vx, vy, vl);

        }

    } else if (1 == inc_x) {

        BLASLONG stride_y = inc_y * sizeof(double);

        for (size_t vl; n > 0; n -= vl, x += vl, y += vl*inc_y) {
            vl = __riscv_vsetvl_e64m8(n);







            vfloat64m8_t vx = __riscv_vle64_v_f64m8(x, vl);
            vfloat64m8_t vy = __riscv_vlse64_v_f64m8(y, stride_y, vl);

            vr = __riscv_vfmacc_vv_f64m8_tu(vr, vx, vy, vl);

        }
    } else if (1 == inc_y) {

        BLASLONG stride_x = inc_x * sizeof(double);

        for (size_t vl; n > 0; n -= vl, x += vl*inc_x, y += vl) {
            vl = __riscv_vsetvl_e64m8(n);







            vfloat64m8_t vx = __riscv_vlse64_v_f64m8(x, stride_x, vl);
            vfloat64m8_t vy = __riscv_vle64_v_f64m8(y, vl);

            vr = __riscv_vfmacc_vv_f64m8_tu(vr, vx, vy, vl);

        }
    } else {

        BLASLONG stride_x = inc_x * sizeof(double);
        BLASLONG stride_y = inc_y * sizeof(double);

        for (size_t vl; n > 0; n -= vl, x += vl*inc_x, y += vl*inc_y) {
            vl = __riscv_vsetvl_e64m8(n);







            vfloat64m8_t vx = __riscv_vlse64_v_f64m8(x, stride_x, vl);
            vfloat64m8_t vy = __riscv_vlse64_v_f64m8(y, stride_y, vl);

            vr = __riscv_vfmacc_vv_f64m8_tu(vr, vx, vy, vl);

        }
    }

    vfloat64m1_t vec_zero = __riscv_vfmv_v_f_f64m1(0, vlmax);
    vfloat64m1_t vec_sum = __riscv_vfredusum_vs_f64m8_f64m1(vr, vec_zero, vlmax);
    dot = __riscv_vfmv_f_s_f64m1_f64(vec_sum);

    return(dot);
}
