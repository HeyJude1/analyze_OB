#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <sys/time.h>
#include <riscv_vector.h>
#include"common.h"
# 29 "zzz_gemmkernel_rvv_v1x8_rvv_template.c" 2
# 46 "zzz_gemmkernel_rvv_v1x8_rvv_template.c"
int dgemm_kernel_rvv(BLASLONG bm, BLASLONG bn, BLASLONG bk, double alpha, double* ba, double* bb, double* C, BLASLONG ldc



  )
{
    BLASLONG i,j,k;
    double *C0,*C1,*C2,*C3,*C4,*C5,*C6,*C7;
    double *ptrba,*ptrbb;



    vfloat64m2_t va0, va1, va2, va3, va4, va5, va6, va7;
    vfloat64m2_t vres0, vres1, vres2, vres3, vres4, vres5, vres6, vres7;
    size_t vl;


    for (j = bn/8; j > 0; j--) {
        C0 = C;
        C1 = C0 + ldc;
        C2 = C1 + ldc;
        C3 = C2 + ldc;
        C4 = C3 + ldc;
        C5 = C4 + ldc;
        C6 = C5 + ldc;
        C7 = C6 + ldc;
        ptrba = ba;

        for (i = bm; i > 0; i -= vl) {
            vl = __riscv_vsetvl_e64m2(i);

            ptrbb = bb;

            vres0 = __riscv_vfmv_v_f_f64m2(0.0, vl);
            vres1 = __riscv_vfmv_v_f_f64m2(0.0, vl);
            vres2 = __riscv_vfmv_v_f_f64m2(0.0, vl);
            vres3 = __riscv_vfmv_v_f_f64m2(0.0, vl);
            vres4 = __riscv_vfmv_v_f_f64m2(0.0, vl);
            vres5 = __riscv_vfmv_v_f_f64m2(0.0, vl);
            vres6 = __riscv_vfmv_v_f_f64m2(0.0, vl);
            vres7 = __riscv_vfmv_v_f_f64m2(0.0, vl);
# 105 "zzz_gemmkernel_rvv_v1x8_rvv_template.c"
            for (k = bk/8; k > 0; k--) {
                va0 = __riscv_vle64_v_f64m2(ptrba, vl);
                ptrba += vl;
                va1 = __riscv_vle64_v_f64m2(ptrba, vl);
                ptrba += vl;

                vres0 = __riscv_vfmacc_vf_f64m2(vres0, *(ptrbb + 0), va0, vl);
                vres1 = __riscv_vfmacc_vf_f64m2(vres1, *(ptrbb + 1), va0, vl);
                vres2 = __riscv_vfmacc_vf_f64m2(vres2, *(ptrbb + 2), va0, vl);
                vres3 = __riscv_vfmacc_vf_f64m2(vres3, *(ptrbb + 3), va0, vl);
                vres4 = __riscv_vfmacc_vf_f64m2(vres4, *(ptrbb + 4), va0, vl);
                vres5 = __riscv_vfmacc_vf_f64m2(vres5, *(ptrbb + 5), va0, vl);
                vres6 = __riscv_vfmacc_vf_f64m2(vres6, *(ptrbb + 6), va0, vl);
                vres7 = __riscv_vfmacc_vf_f64m2(vres7, *(ptrbb + 7), va0, vl);
                ptrbb += 8;

                va2 = __riscv_vle64_v_f64m2(ptrba, vl);
                ptrba += vl;

                vres0 = __riscv_vfmacc_vf_f64m2(vres0, *(ptrbb + 0), va1, vl);
                vres1 = __riscv_vfmacc_vf_f64m2(vres1, *(ptrbb + 1), va1, vl);
                vres2 = __riscv_vfmacc_vf_f64m2(vres2, *(ptrbb + 2), va1, vl);
                vres3 = __riscv_vfmacc_vf_f64m2(vres3, *(ptrbb + 3), va1, vl);
                vres4 = __riscv_vfmacc_vf_f64m2(vres4, *(ptrbb + 4), va1, vl);
                vres5 = __riscv_vfmacc_vf_f64m2(vres5, *(ptrbb + 5), va1, vl);
                vres6 = __riscv_vfmacc_vf_f64m2(vres6, *(ptrbb + 6), va1, vl);
                vres7 = __riscv_vfmacc_vf_f64m2(vres7, *(ptrbb + 7), va1, vl);
                ptrbb += 8;

                va3 = __riscv_vle64_v_f64m2(ptrba, vl);
                ptrba += vl;

                vres0 = __riscv_vfmacc_vf_f64m2(vres0, *(ptrbb + 0), va2, vl);
                vres1 = __riscv_vfmacc_vf_f64m2(vres1, *(ptrbb + 1), va2, vl);
                vres2 = __riscv_vfmacc_vf_f64m2(vres2, *(ptrbb + 2), va2, vl);
                vres3 = __riscv_vfmacc_vf_f64m2(vres3, *(ptrbb + 3), va2, vl);
                vres4 = __riscv_vfmacc_vf_f64m2(vres4, *(ptrbb + 4), va2, vl);
                vres5 = __riscv_vfmacc_vf_f64m2(vres5, *(ptrbb + 5), va2, vl);
                vres6 = __riscv_vfmacc_vf_f64m2(vres6, *(ptrbb + 6), va2, vl);
                vres7 = __riscv_vfmacc_vf_f64m2(vres7, *(ptrbb + 7), va2, vl);
                ptrbb += 8;

                va4 = __riscv_vle64_v_f64m2(ptrba, vl);
                ptrba += vl;

                vres0 = __riscv_vfmacc_vf_f64m2(vres0, *(ptrbb + 0), va3, vl);
                vres1 = __riscv_vfmacc_vf_f64m2(vres1, *(ptrbb + 1), va3, vl);
                vres2 = __riscv_vfmacc_vf_f64m2(vres2, *(ptrbb + 2), va3, vl);
                vres3 = __riscv_vfmacc_vf_f64m2(vres3, *(ptrbb + 3), va3, vl);
                vres4 = __riscv_vfmacc_vf_f64m2(vres4, *(ptrbb + 4), va3, vl);
                vres5 = __riscv_vfmacc_vf_f64m2(vres5, *(ptrbb + 5), va3, vl);
                vres6 = __riscv_vfmacc_vf_f64m2(vres6, *(ptrbb + 6), va3, vl);
                vres7 = __riscv_vfmacc_vf_f64m2(vres7, *(ptrbb + 7), va3, vl);
                ptrbb += 8;

                va5 = __riscv_vle64_v_f64m2(ptrba, vl);
                ptrba += vl;

                vres0 = __riscv_vfmacc_vf_f64m2(vres0, *(ptrbb + 0), va4, vl);
                vres1 = __riscv_vfmacc_vf_f64m2(vres1, *(ptrbb + 1), va4, vl);
                vres2 = __riscv_vfmacc_vf_f64m2(vres2, *(ptrbb + 2), va4, vl);
                vres3 = __riscv_vfmacc_vf_f64m2(vres3, *(ptrbb + 3), va4, vl);
                vres4 = __riscv_vfmacc_vf_f64m2(vres4, *(ptrbb + 4), va4, vl);
                vres5 = __riscv_vfmacc_vf_f64m2(vres5, *(ptrbb + 5), va4, vl);
                vres6 = __riscv_vfmacc_vf_f64m2(vres6, *(ptrbb + 6), va4, vl);
                vres7 = __riscv_vfmacc_vf_f64m2(vres7, *(ptrbb + 7), va4, vl);
                ptrbb += 8;

                va6 = __riscv_vle64_v_f64m2(ptrba, vl);
                ptrba += vl;

                vres0 = __riscv_vfmacc_vf_f64m2(vres0, *(ptrbb + 0), va5, vl);
                vres1 = __riscv_vfmacc_vf_f64m2(vres1, *(ptrbb + 1), va5, vl);
                vres2 = __riscv_vfmacc_vf_f64m2(vres2, *(ptrbb + 2), va5, vl);
                vres3 = __riscv_vfmacc_vf_f64m2(vres3, *(ptrbb + 3), va5, vl);
                vres4 = __riscv_vfmacc_vf_f64m2(vres4, *(ptrbb + 4), va5, vl);
                vres5 = __riscv_vfmacc_vf_f64m2(vres5, *(ptrbb + 5), va5, vl);
                vres6 = __riscv_vfmacc_vf_f64m2(vres6, *(ptrbb + 6), va5, vl);
                vres7 = __riscv_vfmacc_vf_f64m2(vres7, *(ptrbb + 7), va5, vl);
                ptrbb += 8;

                va7 = __riscv_vle64_v_f64m2(ptrba, vl);
                ptrba += vl;

                vres0 = __riscv_vfmacc_vf_f64m2(vres0, *(ptrbb + 0), va6, vl);
                vres1 = __riscv_vfmacc_vf_f64m2(vres1, *(ptrbb + 1), va6, vl);
                vres2 = __riscv_vfmacc_vf_f64m2(vres2, *(ptrbb + 2), va6, vl);
                vres3 = __riscv_vfmacc_vf_f64m2(vres3, *(ptrbb + 3), va6, vl);
                vres4 = __riscv_vfmacc_vf_f64m2(vres4, *(ptrbb + 4), va6, vl);
                vres5 = __riscv_vfmacc_vf_f64m2(vres5, *(ptrbb + 5), va6, vl);
                vres6 = __riscv_vfmacc_vf_f64m2(vres6, *(ptrbb + 6), va6, vl);
                vres7 = __riscv_vfmacc_vf_f64m2(vres7, *(ptrbb + 7), va6, vl);
                ptrbb += 8;

                vres0 = __riscv_vfmacc_vf_f64m2(vres0, *(ptrbb + 0), va7, vl);
                vres1 = __riscv_vfmacc_vf_f64m2(vres1, *(ptrbb + 1), va7, vl);
                vres2 = __riscv_vfmacc_vf_f64m2(vres2, *(ptrbb + 2), va7, vl);
                vres3 = __riscv_vfmacc_vf_f64m2(vres3, *(ptrbb + 3), va7, vl);
                vres4 = __riscv_vfmacc_vf_f64m2(vres4, *(ptrbb + 4), va7, vl);
                vres5 = __riscv_vfmacc_vf_f64m2(vres5, *(ptrbb + 5), va7, vl);
                vres6 = __riscv_vfmacc_vf_f64m2(vres6, *(ptrbb + 6), va7, vl);
                vres7 = __riscv_vfmacc_vf_f64m2(vres7, *(ptrbb + 7), va7, vl);
                ptrbb += 8;
            }


            for (k = bk&7; k > 0; k--) {
                va0 = __riscv_vle64_v_f64m2(ptrba, vl);

                vres0 = __riscv_vfmacc_vf_f64m2(vres0, *(ptrbb + 0), va0, vl);
                vres1 = __riscv_vfmacc_vf_f64m2(vres1, *(ptrbb + 1), va0, vl);
                vres2 = __riscv_vfmacc_vf_f64m2(vres2, *(ptrbb + 2), va0, vl);
                vres3 = __riscv_vfmacc_vf_f64m2(vres3, *(ptrbb + 3), va0, vl);
                vres4 = __riscv_vfmacc_vf_f64m2(vres4, *(ptrbb + 4), va0, vl);
                vres5 = __riscv_vfmacc_vf_f64m2(vres5, *(ptrbb + 5), va0, vl);
                vres6 = __riscv_vfmacc_vf_f64m2(vres6, *(ptrbb + 6), va0, vl);
                vres7 = __riscv_vfmacc_vf_f64m2(vres7, *(ptrbb + 7), va0, vl);

                ptrbb += 8;
                ptrba += vl;
            }

            va0 = __riscv_vle64_v_f64m2(C0, vl);
            va0 = __riscv_vfmacc_vf_f64m2(va0, alpha, vres0, vl);
            __riscv_vse64_v_f64m2(C0, va0, vl);

            va1 = __riscv_vle64_v_f64m2(C1, vl);
            va1 = __riscv_vfmacc_vf_f64m2(va1, alpha, vres1, vl);
            __riscv_vse64_v_f64m2(C1, va1, vl);

            va2 = __riscv_vle64_v_f64m2(C2, vl);
            va2 = __riscv_vfmacc_vf_f64m2(va2, alpha, vres2, vl);
            __riscv_vse64_v_f64m2(C2, va2, vl);

            va3 = __riscv_vle64_v_f64m2(C3, vl);
            va3 = __riscv_vfmacc_vf_f64m2(va3, alpha, vres3, vl);
            __riscv_vse64_v_f64m2(C3, va3, vl);

            va4 = __riscv_vle64_v_f64m2(C4, vl);
            va4 = __riscv_vfmacc_vf_f64m2(va4, alpha, vres4, vl);
            __riscv_vse64_v_f64m2(C4, va4, vl);

            va5 = __riscv_vle64_v_f64m2(C5, vl);
            va5 = __riscv_vfmacc_vf_f64m2(va5, alpha, vres5, vl);
            __riscv_vse64_v_f64m2(C5, va5, vl);

            va6 = __riscv_vle64_v_f64m2(C6, vl);
            va6 = __riscv_vfmacc_vf_f64m2(va6, alpha, vres6, vl);
            __riscv_vse64_v_f64m2(C6, va6, vl);

            va7 = __riscv_vle64_v_f64m2(C7, vl);
            va7 = __riscv_vfmacc_vf_f64m2(va7, alpha, vres7, vl);
            __riscv_vse64_v_f64m2(C7, va7, vl);

            C0 += vl;
            C1 += vl;
            C2 += vl;
            C3 += vl;
            C4 += vl;
            C5 += vl;
            C6 += vl;
            C7 += vl;
        }

        bb += (bk<<3);
        C += (ldc<<3);
    }


    if (bn & 4) {
        C0 = C;
        C1 = C0 + ldc;
        C2 = C1 + ldc;
        C3 = C2 + ldc;
        ptrba = ba;

        for (i = bm; i > 0; i -= vl) {
            vl = __riscv_vsetvl_e64m2(i);

            ptrbb = bb;

            vres0 = __riscv_vfmv_v_f_f64m2(0.0, vl);
            vres1 = __riscv_vfmv_v_f_f64m2(0.0, vl);
            vres2 = __riscv_vfmv_v_f_f64m2(0.0, vl);
            vres3 = __riscv_vfmv_v_f_f64m2(0.0, vl);
# 305 "zzz_gemmkernel_rvv_v1x8_rvv_template.c"
            for (k = bk/8; k > 0; k--) {
                va0 = __riscv_vle64_v_f64m2(ptrba, vl);
                ptrba += vl;
                va1 = __riscv_vle64_v_f64m2(ptrba, vl);
                ptrba += vl;

                vres0 = __riscv_vfmacc_vf_f64m2(vres0, *(ptrbb + 0), va0, vl);
                vres1 = __riscv_vfmacc_vf_f64m2(vres1, *(ptrbb + 1), va0, vl);
                vres2 = __riscv_vfmacc_vf_f64m2(vres2, *(ptrbb + 2), va0, vl);
                vres3 = __riscv_vfmacc_vf_f64m2(vres3, *(ptrbb + 3), va0, vl);
                ptrbb += 4;
                va2 = __riscv_vle64_v_f64m2(ptrba, vl);
                ptrba += vl;

                vres0 = __riscv_vfmacc_vf_f64m2(vres0, *(ptrbb + 0), va1, vl);
                vres1 = __riscv_vfmacc_vf_f64m2(vres1, *(ptrbb + 1), va1, vl);
                vres2 = __riscv_vfmacc_vf_f64m2(vres2, *(ptrbb + 2), va1, vl);
                vres3 = __riscv_vfmacc_vf_f64m2(vres3, *(ptrbb + 3), va1, vl);
                ptrbb += 4;
                va3 = __riscv_vle64_v_f64m2(ptrba, vl);
                ptrba += vl;

                vres0 = __riscv_vfmacc_vf_f64m2(vres0, *(ptrbb + 0), va2, vl);
                vres1 = __riscv_vfmacc_vf_f64m2(vres1, *(ptrbb + 1), va2, vl);
                vres2 = __riscv_vfmacc_vf_f64m2(vres2, *(ptrbb + 2), va2, vl);
                vres3 = __riscv_vfmacc_vf_f64m2(vres3, *(ptrbb + 3), va2, vl);
                ptrbb += 4;
                va4 = __riscv_vle64_v_f64m2(ptrba, vl);
                ptrba += vl;

                vres0 = __riscv_vfmacc_vf_f64m2(vres0, *(ptrbb + 0), va3, vl);
                vres1 = __riscv_vfmacc_vf_f64m2(vres1, *(ptrbb + 1), va3, vl);
                vres2 = __riscv_vfmacc_vf_f64m2(vres2, *(ptrbb + 2), va3, vl);
                vres3 = __riscv_vfmacc_vf_f64m2(vres3, *(ptrbb + 3), va3, vl);
                ptrbb += 4;
                va5 = __riscv_vle64_v_f64m2(ptrba, vl);
                ptrba += vl;

                vres0 = __riscv_vfmacc_vf_f64m2(vres0, *(ptrbb + 0), va4, vl);
                vres1 = __riscv_vfmacc_vf_f64m2(vres1, *(ptrbb + 1), va4, vl);
                vres2 = __riscv_vfmacc_vf_f64m2(vres2, *(ptrbb + 2), va4, vl);
                vres3 = __riscv_vfmacc_vf_f64m2(vres3, *(ptrbb + 3), va4, vl);
                ptrbb += 4;
                va6 = __riscv_vle64_v_f64m2(ptrba, vl);
                ptrba += vl;

                vres0 = __riscv_vfmacc_vf_f64m2(vres0, *(ptrbb + 0), va5, vl);
                vres1 = __riscv_vfmacc_vf_f64m2(vres1, *(ptrbb + 1), va5, vl);
                vres2 = __riscv_vfmacc_vf_f64m2(vres2, *(ptrbb + 2), va5, vl);
                vres3 = __riscv_vfmacc_vf_f64m2(vres3, *(ptrbb + 3), va5, vl);
                ptrbb += 4;
                va7 = __riscv_vle64_v_f64m2(ptrba, vl);
                ptrba += vl;

                vres0 = __riscv_vfmacc_vf_f64m2(vres0, *(ptrbb + 0), va6, vl);
                vres1 = __riscv_vfmacc_vf_f64m2(vres1, *(ptrbb + 1), va6, vl);
                vres2 = __riscv_vfmacc_vf_f64m2(vres2, *(ptrbb + 2), va6, vl);
                vres3 = __riscv_vfmacc_vf_f64m2(vres3, *(ptrbb + 3), va6, vl);
                ptrbb += 4;

                vres0 = __riscv_vfmacc_vf_f64m2(vres0, *(ptrbb + 0), va7, vl);
                vres1 = __riscv_vfmacc_vf_f64m2(vres1, *(ptrbb + 1), va7, vl);
                vres2 = __riscv_vfmacc_vf_f64m2(vres2, *(ptrbb + 2), va7, vl);
                vres3 = __riscv_vfmacc_vf_f64m2(vres3, *(ptrbb + 3), va7, vl);
                ptrbb += 4;
            }


            for (k = bk&7; k > 0; k--) {
                va0 = __riscv_vle64_v_f64m2(ptrba, vl);

                vres0 = __riscv_vfmacc_vf_f64m2(vres0, *(ptrbb + 0), va0, vl);
                vres1 = __riscv_vfmacc_vf_f64m2(vres1, *(ptrbb + 1), va0, vl);
                vres2 = __riscv_vfmacc_vf_f64m2(vres2, *(ptrbb + 2), va0, vl);
                vres3 = __riscv_vfmacc_vf_f64m2(vres3, *(ptrbb + 3), va0, vl);

                ptrbb += 4;
                ptrba += vl;
            }

            va0 = __riscv_vle64_v_f64m2(C0, vl);
            va0 = __riscv_vfmacc_vf_f64m2(va0, alpha, vres0, vl);
            __riscv_vse64_v_f64m2(C0, va0, vl);

            va1 = __riscv_vle64_v_f64m2(C1, vl);
            va1 = __riscv_vfmacc_vf_f64m2(va1, alpha, vres1, vl);
            __riscv_vse64_v_f64m2(C1, va1, vl);

            va2 = __riscv_vle64_v_f64m2(C2, vl);
            va2 = __riscv_vfmacc_vf_f64m2(va2, alpha, vres2, vl);
            __riscv_vse64_v_f64m2(C2, va2, vl);

            va3 = __riscv_vle64_v_f64m2(C3, vl);
            va3 = __riscv_vfmacc_vf_f64m2(va3, alpha, vres3, vl);
            __riscv_vse64_v_f64m2(C3, va3, vl);

            C0 += vl;
            C1 += vl;
            C2 += vl;
            C3 += vl;
        }

        bb += (bk<<2);
        C += (ldc<<2);
    }


    if (bn & 2) {
        C0 = C;
        C1 = C0 + ldc;
        ptrba = ba;

        for (i = bm; i > 0; i -= vl) {
            vl = __riscv_vsetvl_e64m2(i);

            ptrbb = bb;

            vres0 = __riscv_vfmv_v_f_f64m2(0.0, vl);
            vres1 = __riscv_vfmv_v_f_f64m2(0.0, vl);
# 436 "zzz_gemmkernel_rvv_v1x8_rvv_template.c"
            for (k = bk/8; k > 0; k--) {
                va0 = __riscv_vle64_v_f64m2(ptrba, vl);
                ptrba += vl;
                va1 = __riscv_vle64_v_f64m2(ptrba, vl);
                ptrba += vl;

                vres0 = __riscv_vfmacc_vf_f64m2(vres0, *(ptrbb + 0), va0, vl);
                vres1 = __riscv_vfmacc_vf_f64m2(vres1, *(ptrbb + 1), va0, vl);
                ptrbb += 2;
                va2 = __riscv_vle64_v_f64m2(ptrba, vl);
                ptrba += vl;

                vres0 = __riscv_vfmacc_vf_f64m2(vres0, *(ptrbb + 0), va1, vl);
                vres1 = __riscv_vfmacc_vf_f64m2(vres1, *(ptrbb + 1), va1, vl);
                ptrbb += 2;
                va3 = __riscv_vle64_v_f64m2(ptrba, vl);
                ptrba += vl;

                vres0 = __riscv_vfmacc_vf_f64m2(vres0, *(ptrbb + 0), va2, vl);
                vres1 = __riscv_vfmacc_vf_f64m2(vres1, *(ptrbb + 1), va2, vl);
                ptrbb += 2;
                va4 = __riscv_vle64_v_f64m2(ptrba, vl);
                ptrba += vl;

                vres0 = __riscv_vfmacc_vf_f64m2(vres0, *(ptrbb + 0), va3, vl);
                vres1 = __riscv_vfmacc_vf_f64m2(vres1, *(ptrbb + 1), va3, vl);
                ptrbb += 2;
                va5 = __riscv_vle64_v_f64m2(ptrba, vl);
                ptrba += vl;

                vres0 = __riscv_vfmacc_vf_f64m2(vres0, *(ptrbb + 0), va4, vl);
                vres1 = __riscv_vfmacc_vf_f64m2(vres1, *(ptrbb + 1), va4, vl);
                ptrbb += 2;
                va6 = __riscv_vle64_v_f64m2(ptrba, vl);
                ptrba += vl;

                vres0 = __riscv_vfmacc_vf_f64m2(vres0, *(ptrbb + 0), va5, vl);
                vres1 = __riscv_vfmacc_vf_f64m2(vres1, *(ptrbb + 1), va5, vl);
                ptrbb += 2;
                va7 = __riscv_vle64_v_f64m2(ptrba, vl);
                ptrba += vl;

                vres0 = __riscv_vfmacc_vf_f64m2(vres0, *(ptrbb + 0), va6, vl);
                vres1 = __riscv_vfmacc_vf_f64m2(vres1, *(ptrbb + 1), va6, vl);
                ptrbb += 2;

                vres0 = __riscv_vfmacc_vf_f64m2(vres0, *(ptrbb + 0), va7, vl);
                vres1 = __riscv_vfmacc_vf_f64m2(vres1, *(ptrbb + 1), va7, vl);
                ptrbb += 2;
            }


            for (k = bk&7; k > 0; k--) {
                va0 = __riscv_vle64_v_f64m2(ptrba, vl);

                vres0 = __riscv_vfmacc_vf_f64m2(vres0, *(ptrbb + 0), va0, vl);
                vres1 = __riscv_vfmacc_vf_f64m2(vres1, *(ptrbb + 1), va0, vl);

                ptrbb += 2;
                ptrba += vl;
            }

            va0 = __riscv_vle64_v_f64m2(C0, vl);
            va0 = __riscv_vfmacc_vf_f64m2(va0, alpha, vres0, vl);
            __riscv_vse64_v_f64m2(C0, va0, vl);

            va1 = __riscv_vle64_v_f64m2(C1, vl);
            va1 = __riscv_vfmacc_vf_f64m2(va1, alpha, vres1, vl);
            __riscv_vse64_v_f64m2(C1, va1, vl);

            C0 += vl;
            C1 += vl;
        }

        bb += (bk<<1);
        C += (ldc<<1);
    }


    if (bn & 1) {
        C0 = C;
        ptrba = ba;

        for (i = bm; i > 0; i -= vl) {
            vl = __riscv_vsetvl_e64m2(i);

            ptrbb = bb;

            vres0 = __riscv_vfmv_v_f_f64m2(0.0, vl);
# 536 "zzz_gemmkernel_rvv_v1x8_rvv_template.c"
            for (k = bk/8; k > 0; k--) {
                va0 = __riscv_vle64_v_f64m2(ptrba, vl);
                ptrba += vl;
                va1 = __riscv_vle64_v_f64m2(ptrba, vl);
                ptrba += vl;

                vres0 = __riscv_vfmacc_vf_f64m2(vres0, *(ptrbb + 0), va0, vl);
                ptrbb += 1;
                va2 = __riscv_vle64_v_f64m2(ptrba, vl);
                ptrba += vl;

                vres0 = __riscv_vfmacc_vf_f64m2(vres0, *(ptrbb + 0), va1, vl);
                ptrbb += 1;
                va3 = __riscv_vle64_v_f64m2(ptrba, vl);
                ptrba += vl;

                vres0 = __riscv_vfmacc_vf_f64m2(vres0, *(ptrbb + 0), va2, vl);
                ptrbb += 1;
                va4 = __riscv_vle64_v_f64m2(ptrba, vl);
                ptrba += vl;

                vres0 = __riscv_vfmacc_vf_f64m2(vres0, *(ptrbb + 0), va3, vl);
                ptrbb += 1;
                va5 = __riscv_vle64_v_f64m2(ptrba, vl);
                ptrba += vl;

                vres0 = __riscv_vfmacc_vf_f64m2(vres0, *(ptrbb + 0), va4, vl);
                ptrbb += 1;
                va6 = __riscv_vle64_v_f64m2(ptrba, vl);
                ptrba += vl;

                vres0 = __riscv_vfmacc_vf_f64m2(vres0, *(ptrbb + 0), va5, vl);
                ptrbb += 1;
                va7 = __riscv_vle64_v_f64m2(ptrba, vl);
                ptrba += vl;

                vres0 = __riscv_vfmacc_vf_f64m2(vres0, *(ptrbb + 0), va6, vl);
                ptrbb += 1;

                vres0 = __riscv_vfmacc_vf_f64m2(vres0, *(ptrbb + 0), va7, vl);
                ptrbb += 1;
            }


            for (k = bk&7; k > 0; k--) {
                va0 = __riscv_vle64_v_f64m2(ptrba, vl);

                vres0 = __riscv_vfmacc_vf_f64m2(vres0, *(ptrbb + 0), va0, vl);

                ptrbb += 1;
                ptrba += vl;
            }

            va0 = __riscv_vle64_v_f64m2(C0, vl);
            va0 = __riscv_vfmacc_vf_f64m2(va0, alpha, vres0, vl);
            __riscv_vse64_v_f64m2(C0, va0, vl);

            C0 += vl;
        }

        bb += (bk);
        C += (ldc);
    }

    return 0;
}
