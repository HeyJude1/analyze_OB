#include <stdio.h>
#include"common.h"
# 29 "zzz_daxpy_k_template.c" 2


# 30 "zzz_daxpy_k_template.c"
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
