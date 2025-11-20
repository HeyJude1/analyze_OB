# 38 "../kernel/x86_64/../arm/dot.c" 2
double ddot_k(BLASLONG n, double *x, BLASLONG inc_x, double *y, BLASLONG inc_y)
{
 BLASLONG i=0;
 BLASLONG ix=0,iy=0;
 double dot = 0.0 ;
 if ( n < 0 ) return(dot);
 while(i < n)
 {
  dot += y[iy] * x[ix] ;
  ix += inc_x ;
  iy += inc_y ;
  i++ ;
 }
 return(dot);
}
