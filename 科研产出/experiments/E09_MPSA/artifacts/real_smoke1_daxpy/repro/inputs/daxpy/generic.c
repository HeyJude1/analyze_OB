/* 由 tools/clean_generic.py 从 OpenBLAS 提取的 generic.c
 * 来源：13_daxpy_op\daxpy_k.c
 * 清洗：去掉 common.h 依赖，仅保留最小 typedef；函数体保持原样（未优化参考实现）。
 */
#include <stdio.h>
#include <stddef.h>
#include <stdlib.h>
#include <math.h>

typedef long long BLASLONG;
typedef int blasint;

#include <stdio.h>

int daxpy_k(BLASLONG n, BLASLONG dummy0, BLASLONG dummy1, double da, double *x, BLASLONG inc_x, double *y, BLASLONG inc_y, double *dummy, BLASLONG dummy2)
{
 BLASLONG i=0;
 BLASLONG ix,iy;

 if ( n < 0 ) return(0);
 if ( da == 0.0 ) return(0);

 ix = 0;
 iy = 0;

 while(i < n)
 {

  y[iy] += da * x[ix] ;
  ix += inc_x ;
  iy += inc_y ;
  i++ ;

 }
 return(0);

}
