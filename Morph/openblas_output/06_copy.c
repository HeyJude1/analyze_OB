# 38 "../kernel/x86_64/../arm/copy.c" 2
int dcopy_k(BLASLONG n, double *x, BLASLONG inc_x, double *y, BLASLONG inc_y)
{
 BLASLONG i=0;
 BLASLONG ix=0,iy=0;
 if ( n < 0 ) return(0);
 while(i < n)
 {
  y[iy] = x[ix] ;
  ix += inc_x ;
  iy += inc_y ;
  i++ ;
 }
 return(0);
}
