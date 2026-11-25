#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <sys/time.h>
#include <riscv_vector.h>
#include"common.h"
# 29 "zzz_daxpy_k_rvv_template.c" 2
# 58 "zzz_daxpy_k_rvv_template.c"
int daxpy_k_rvv(BLASLONG n, BLASLONG dummy0, BLASLONG dummy1, double da, double *x, BLASLONG inc_x, double *y, BLASLONG inc_y, double *dummy, BLASLONG dummy2)
{
    if ( n <= 0 ) return(0);
    if ( da == 0.0 ) return(0);

    vfloat64m8_t vx, vy;

    if(inc_x == 1 && inc_y == 1) {

        for (size_t vl; n > 0; n -= vl, x += vl, y += vl) {
            vl = __riscv_vsetvl_e64m8(n);

            vx = __riscv_vle64_v_f64m8(x, vl);
            vy = __riscv_vle64_v_f64m8(y, vl);
            vy = __riscv_vfmacc_vf_f64m8(vy, da, vx, vl);
            __riscv_vse64_v_f64m8 (y, vy, vl);
        }

    } else if (1 == inc_y) {

        BLASLONG stride_x = inc_x * sizeof(double);

        for (size_t vl; n > 0; n -= vl, x += vl*inc_x, y += vl) {
            vl = __riscv_vsetvl_e64m8(n);

            vx = __riscv_vlse64_v_f64m8(x, stride_x, vl);
            vy = __riscv_vle64_v_f64m8(y, vl);
            vy = __riscv_vfmacc_vf_f64m8(vy, da, vx, vl);
            __riscv_vse64_v_f64m8(y, vy, vl);
        }

    } else if (1 == inc_x && 0 != inc_y) {

        BLASLONG stride_y = inc_y * sizeof(double);

        for (size_t vl; n > 0; n -= vl, x += vl, y += vl*inc_y) {
            vl = __riscv_vsetvl_e64m8(n);

            vx = __riscv_vle64_v_f64m8(x, vl);
            vy = __riscv_vlse64_v_f64m8(y, stride_y, vl);
            vy = __riscv_vfmacc_vf_f64m8(vy, da, vx, vl);
            __riscv_vsse64_v_f64m8(y, stride_y, vy, vl);
        }

    } else if( 0 == inc_y ) {
        BLASLONG stride_x = inc_x * sizeof(double);
        size_t in_vl = __riscv_vsetvl_e64m8(n);
        vy = __riscv_vfmv_v_f_f64m8( y[0], in_vl );

        for (size_t vl; n > 0; n -= vl, x += vl*inc_x) {
            vl = __riscv_vsetvl_e64m8(n);
            vx = __riscv_vlse64_v_f64m8(x, stride_x, vl);
            vy = __riscv_vfmacc_vf_f64m8(vy, da, vx, vl);
        }
        vfloat64m1_t vres = __riscv_vfmv_v_f_f64m1( 0.0f, 1 );
        vres = __riscv_vfredusum_vs_f64m8_f64m1( vy, vres, in_vl );
        __riscv_vse64_v_f64m1(y, vres, 1);
    } else {
        BLASLONG stride_x = inc_x * sizeof(double);
        BLASLONG stride_y = inc_y * sizeof(double);

        for (size_t vl; n > 0; n -= vl, x += vl*inc_x, y += vl*inc_y) {
            vl = __riscv_vsetvl_e64m8(n);

            vx = __riscv_vlse64_v_f64m8(x, stride_x, vl);
            vy = __riscv_vlse64_v_f64m8(y, stride_y, vl);
            vy = __riscv_vfmacc_vf_f64m8(vy, da, vx, vl);
            __riscv_vsse64_v_f64m8(y, stride_y, vy, vl);
        }

    }

    return(0);
}
