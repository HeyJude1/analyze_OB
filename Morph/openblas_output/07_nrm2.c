# 38 "../kernel/x86_64/../arm/nrm2.c" 2
# 52 "../kernel/x86_64/../arm/nrm2.c"
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
