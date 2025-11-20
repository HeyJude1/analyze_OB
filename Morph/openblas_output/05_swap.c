# 36 "../kernel/x86_64/../arm/swap.c" 2
int dswap_k(BLASLONG n, BLASLONG dummy0, BLASLONG dummy1, double dummy3, double *x, BLASLONG inc_x, double *y, BLASLONG inc_y, double *dummy, BLASLONG dummy2)
{
 BLASLONG i=0;
 BLASLONG ix=0,iy=0;
 double temp;
 if ( n < 0 ) return(0);
 while(i < n)
 {
  temp = x[ix] ;
  x[ix] = y[iy] ;
  y[iy] = temp ;
  ix += inc_x ;
  iy += inc_y ;
  i++ ;
 }
 return(0);
}
