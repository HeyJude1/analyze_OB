/* 由 tools/clean_generic.py 从 OpenBLAS 提取的 generic.c
 * 来源：37_dtrmm_op1\dtrmm_k.c
 * 清洗：去掉 common.h 依赖，仅保留最小 typedef；函数体保持原样（未优化参考实现）。
 */
#include <stdio.h>
#include <stddef.h>
#include <stdlib.h>
#include <math.h>

typedef long long BLASLONG;
typedef int blasint;

#include <stdio.h>

int dtrmm_kernel_LN(BLASLONG bm,BLASLONG bn,BLASLONG bk,double alpha,double* ba,double* bb,double* C,BLASLONG ldc

  ,BLASLONG offset

  )
{
   BLASLONG i,j,k;
   double *C0,*C1,*ptrba,*ptrbb;
   double res0,res1,res2,res3,load0,load1,load2,load3,load4,load5,load6,load7;
   BLASLONG off, temp;



   off = 0;

   for (j=0; j<bn/2; j+=1)
     {
        C0 = C;
        C1 = C0+ldc;

  off = offset;

        ptrba = ba;
        for (i=0; i<bm/2; i+=1)
          {




     ptrba += off*2;
     ptrbb = bb + off*2;

             res0 = 0;
             res1 = 0;
             res2 = 0;
             res3 = 0;


    temp = bk-off;





             for (k=0; k<temp/4; k+=1)
               {
                  load0 = ptrba[2*0+0];
                  load1 = ptrbb[2*0+0];
                  res0 = res0+load0*load1;
                  load2 = ptrba[2*0+1];
                  res1 = res1+load2*load1;
                  load3 = ptrbb[2*0+1];
                  res2 = res2+load0*load3;
                  res3 = res3+load2*load3;
                  load4 = ptrba[2*1+0];
                  load5 = ptrbb[2*1+0];
                  res0 = res0+load4*load5;
                  load6 = ptrba[2*1+1];
                  res1 = res1+load6*load5;
                  load7 = ptrbb[2*1+1];
                  res2 = res2+load4*load7;
                  res3 = res3+load6*load7;
                  load0 = ptrba[2*2+0];
                  load1 = ptrbb[2*2+0];
                  res0 = res0+load0*load1;
                  load2 = ptrba[2*2+1];
                  res1 = res1+load2*load1;
                  load3 = ptrbb[2*2+1];
                  res2 = res2+load0*load3;
                  res3 = res3+load2*load3;
                  load4 = ptrba[2*3+0];
                  load5 = ptrbb[2*3+0];
                  res0 = res0+load4*load5;
                  load6 = ptrba[2*3+1];
                  res1 = res1+load6*load5;
                  load7 = ptrbb[2*3+1];
                  res2 = res2+load4*load7;
                  res3 = res3+load6*load7;
                  ptrba = ptrba+8;
                  ptrbb = ptrbb+8;
               }
             for (k=0; k<(temp&3); k+=1)
               {
                  load0 = ptrba[2*0+0];
                  load1 = ptrbb[2*0+0];
                  res0 = res0+load0*load1;
                  load2 = ptrba[2*0+1];
                  res1 = res1+load2*load1;
                  load3 = ptrbb[2*0+1];
                  res2 = res2+load0*load3;
                  res3 = res3+load2*load3;
                  ptrba = ptrba+2;
                  ptrbb = ptrbb+2;
               }
             res0 = res0*alpha;
             C0[0] = res0;
             res1 = res1*alpha;
             C0[1] = res1;
             res2 = res2*alpha;
             C1[0] = res2;
             res3 = res3*alpha;
             C1[1] = res3;

    off += 2;

             C0 = C0+2;
             C1 = C1+2;
          }
        for (i=0; i<(bm&1); i+=1)
          {



    ptrba += off;
    ptrbb = bb+off*2;

             res0 = 0;
             res1 = 0;

    temp = bk-off;





             for (k=0; k<temp; k+=1)
               {
                  load0 = ptrba[0+0];
                  load1 = ptrbb[2*0+0];
                  res0 = res0+load0*load1;
                  load2 = ptrbb[2*0+1];
                  res1 = res1+load0*load2;
                  ptrba = ptrba+1;
                  ptrbb = ptrbb+2;
               }
             res0 = res0*alpha;
             C0[0] = res0;
             res1 = res1*alpha;
             C1[0] = res1;

    off += 1;

             C0 = C0+1;
             C1 = C1+1;
          }



        k = (bk<<1);
        bb = bb+k;
        i = (ldc<<1);
        C = C+i;
     }
   for (j=0; j<(bn&1); j+=1)
     {
        C0 = C;

  off = offset;

        ptrba = ba;
        for (i=0; i<bm/2; i+=1)
          {




     ptrba += off*2;
     ptrbb = bb + off;

             res0 = 0;
             res1 = 0;


    temp = bk-off;





    for (k=0; k<temp; k+=1)
               {
                  load0 = ptrba[2*0+0];
                  load1 = ptrbb[0+0];
                  res0 = res0+load0*load1;
                  load2 = ptrba[2*0+1];
                  res1 = res1+load2*load1;
                  ptrba = ptrba+2;
                  ptrbb = ptrbb+1;
               }
             res0 = res0*alpha;
             C0[0] = res0;
             res1 = res1*alpha;
    C0[1] = res1;

    off += 2;


             C0 = C0+2;
          }
        for (i=0; i<(bm&1); i+=1)
          {



    ptrba += off;
    ptrbb = bb+off;

             res0 = 0;

    temp = bk-off;





    for (k=0; k<temp; k+=1)
               {
                  load0 = ptrba[0+0];
                  load1 = ptrbb[0+0];
                  res0 = res0+load0*load1;
                  ptrba = ptrba+1;
                  ptrbb = ptrbb+1;
               }
             res0 = res0*alpha;
             C0[0] = res0;

    off += 1;

             C0 = C0+1;
          }



        k = (bk<<0);
        bb = bb+k;
        C = C+ldc;
     }
   return 0;
}
