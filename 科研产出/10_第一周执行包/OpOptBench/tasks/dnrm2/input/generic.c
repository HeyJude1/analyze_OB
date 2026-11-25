/* 由 tools/clean_generic.py 从 OpenBLAS 提取的 generic.c
 * 来源：22_dnrm2_op\dnrm2_k.c
 * 清洗：去掉 common.h 依赖，仅保留最小 typedef；函数体保持原样（未优化参考实现）。
 */
#include <stdio.h>
#include <stddef.h>
#include <stdlib.h>
#include <math.h>

typedef long long BLASLONG;
typedef int blasint;

#include <stdio.h>

double dnrm2_k(BLASLONG n, double *x, BLASLONG inc_x)
{
 BLASLONG i=0;
 double scale = 0.0;
 double ssq = 1.0;
 double absxi = 0.0;


 if (n <= 0 || inc_x == 0) return(0.0);
 if ( n == 1 ) return( fabs(x[0]) );

 n *= inc_x;
 while(abs(i) < abs(n))
 {

  if ( x[i] != 0.0 )
  {
   absxi = fabs( x[i] );
   if ( scale < absxi )
   {
    ssq = 1 + ssq * ( scale / absxi ) * ( scale / absxi );
    scale = absxi ;
   }
   else
   {
    ssq += ( absxi/scale ) * ( absxi/scale );
   }

  }
  i += inc_x;
 }
 scale = scale * sqrt( ssq );
 return(scale);

}
