/* 由 tools/clean_generic.py 从 OpenBLAS 提取的 generic.c
 * 来源：24_dscal_op\dscal_k.c
 * 清洗：去掉 common.h 依赖，仅保留最小 typedef；函数体保持原样（未优化参考实现）。
 */
#include <stdio.h>
#include <stddef.h>
#include <stdlib.h>
#include <math.h>

typedef long long BLASLONG;
typedef int blasint;

#include <stdio.h>

int dscal_k(BLASLONG n, BLASLONG dummy0, BLASLONG dummy1, double da, double *x, BLASLONG inc_x, double *y, BLASLONG inc_y, double *dummy, BLASLONG dummy2)
{
 BLASLONG i=0,j=0;

 while(j < n)
 {

  if ( da == 0.0 )
   x[i]=0.0;
  else
   x[i] = da * x[i] ;

  i += inc_x ;
  j++;

 }
 return 0;

}
