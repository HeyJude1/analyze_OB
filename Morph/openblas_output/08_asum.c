# 39 "../kernel/x86_64/../arm/asum.c" 2
# 52 "../kernel/x86_64/../arm/asum.c"
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
