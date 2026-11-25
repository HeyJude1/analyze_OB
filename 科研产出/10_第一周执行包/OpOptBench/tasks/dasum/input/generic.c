/* 由 tools/clean_generic.py 从 OpenBLAS 提取的 generic.c
 * 来源：11_dasum_op\dasum_k.c
 * 清洗：去掉 common.h 依赖，仅保留最小 typedef；函数体保持原样（未优化参考实现）。
 */
#include <stdio.h>
#include <stddef.h>
#include <stdlib.h>
#include <math.h>

typedef long long BLASLONG;
typedef int blasint;

#include <stdio.h>

double dasum_k(BLASLONG n, double *x, BLASLONG inc_x)
{
 BLASLONG i=0;
 double sumf = 0.0;
 if (n <= 0 || inc_x <= 0) return(sumf);

 n *= inc_x;
 while(i < n)
 {
  sumf += fabs(x[i]);
  i += inc_x;
 }
 return(sumf);
}
