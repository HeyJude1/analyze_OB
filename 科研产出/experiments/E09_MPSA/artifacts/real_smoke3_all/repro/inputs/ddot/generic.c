/* 由 tools/clean_generic.py 从 OpenBLAS 提取的 generic.c
 * 来源：15_ddot_op\ddot_k.c
 * 清洗：去掉 common.h 依赖，仅保留最小 typedef；函数体保持原样（未优化参考实现）。
 */
#include <stdio.h>
#include <stddef.h>
#include <stdlib.h>
#include <math.h>

typedef long long BLASLONG;
typedef int blasint;

#include <stdio.h>

double ddot_k(BLASLONG n, double *x, BLASLONG inc_x, double *y, BLASLONG inc_y)

{
 BLASLONG i=0;
 BLASLONG ix=0,iy=0;
 double dot = 0.0 ;

 if ( n < 0 ) return(dot);

 while(i < n)
 {



  dot += y[iy] * x[ix];

  ix += inc_x ;
  iy += inc_y ;
  i++ ;

 }
 return(dot);

}
