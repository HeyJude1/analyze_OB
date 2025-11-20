# 38 "../kernel/x86_64/../arm/iamin.c" 2
# 51 "../kernel/x86_64/../arm/iamin.c"
BLASLONG idamin_k(BLASLONG n, double *x, BLASLONG inc_x)
{
 BLASLONG i=0;
 BLASLONG ix=0;
 double minf=0.0;
 BLASLONG min=0;
 if (n <= 0 || inc_x <= 0) return(min);
 minf=fabs(x[0]);
 ix += inc_x;
 i++;
 while(i < n)
 {
  if( fabs(x[ix]) < fabs(minf) )
  {
   min = i;
   minf = fabs(x[ix]);
  }
  ix += inc_x;
  i++;
 }
 return(min+1);
}
