/* 由 tools/clean_generic.py 从 OpenBLAS 提取的 generic.c
 * 来源：12_daxpby_op\daxpby_k.c
 * 清洗：去掉 common.h 依赖，仅保留最小 typedef；函数体保持原样（未优化参考实现）。
 */
#include <stdio.h>
#include <stddef.h>
#include <stdlib.h>
#include <math.h>

typedef long long BLASLONG;
typedef int blasint;

#include <stdio.h>

int daxpby_k(BLASLONG n, double alpha, double *x, BLASLONG inc_x, double beta, double *y, BLASLONG inc_y)
{
 BLASLONG i=0;
 BLASLONG ix,iy;

 if ( n < 0 ) return(0);

 ix = 0;
 iy = 0;

 if ( beta == 0.0 )
 {

  if ( alpha == 0.0 )
  {
   while(i < n)
   {
    y[iy] = 0.0 ;
    iy += inc_y ;
    i++ ;
   }
  }
  else
  {
   while(i < n)
   {
    y[iy] = alpha * x[ix] ;
    ix += inc_x ;
    iy += inc_y ;
    i++ ;
   }


  }

 }
 else
 {

  if ( alpha == 0.0 )
  {
   while(i < n)
   {
    y[iy] = beta * y[iy] ;
    iy += inc_y ;
    i++ ;
   }
  }
  else
  {
   while(i < n)
   {
    y[iy] = alpha * x[ix] + beta * y[iy] ;
    ix += inc_x ;
    iy += inc_y ;
    i++ ;
   }
  }

 }

 return(0);

}
