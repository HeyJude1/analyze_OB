# 38 "../kernel/x86_64/../arm/iamax.c" 2
# 51 "../kernel/x86_64/../arm/iamax.c"
BLASLONG idamax_k(BLASLONG n, double *x, BLASLONG inc_x)
{
 BLASLONG i=0;
 BLASLONG ix=0;
 double maxf=0.0;
 BLASLONG max=0;
 if (n <= 0 || inc_x <= 0) return(max);
 maxf=fabs(x[0]);
 ix += inc_x;
 i++;
 while(i < n)
 {
  if( fabs(x[ix]) > maxf )
  {
   max = i;
   maxf = fabs(x[ix]);
  }
  ix += inc_x;
  i++;
 }
 return(max+1);
}
