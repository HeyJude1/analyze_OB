# 30 "../kernel/x86_64/../arm/axpby.c" 2
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
