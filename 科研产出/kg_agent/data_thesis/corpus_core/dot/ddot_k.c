#include <stdio.h>
#include"common.h"
# 29 "zzz_ddot_k_template.c" 2





# 33 "zzz_ddot_k_template.c"
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
