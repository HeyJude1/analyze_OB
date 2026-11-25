/* 由 tools/clean_generic.py 从 OpenBLAS 提取的 generic.c
 * 来源：17_dgemv_op\dgemv_k.c
 * 清洗：去掉 common.h 依赖，仅保留最小 typedef；函数体保持原样（未优化参考实现）。
 */
#include <stdio.h>
#include <stddef.h>
#include <stdlib.h>
#include <math.h>

typedef long long BLASLONG;
typedef int blasint;

#include <stdio.h>

int dgemv_n(BLASLONG m, BLASLONG n, BLASLONG dummy1, double alpha, double *a, BLASLONG lda, double *x, BLASLONG inc_x, double *y, BLASLONG inc_y, double *buffer)
{
 BLASLONG i;
 BLASLONG ix,iy;
 BLASLONG j;
 double *a_ptr;
 double temp;

 ix = 0;
 a_ptr = a;

 for (j=0; j<n; j++)
 {
  temp = alpha * x[ix];
  iy = 0;
  for (i=0; i<m; i++)
  {
   y[iy] += temp * a_ptr[i];
   iy += inc_y;
  }
  a_ptr += lda;
  ix += inc_x;
 }
 return(0);
}
