# 40 "../kernel/x86_64/../arm/gemv_n.c" 2
int dgemv_n(BLASLONG m, BLASLONG n, BLASLONG dummy1, double alpha, double *a, BLASLONG lda, double *x, BLASLONG inc_x, double *y, BLASLONG inc_y, double *buffer)
{
 BLASLONG i;
 BLASLONG ix,iy;
 BLASLONG j;
 double *a_ptr;
 double temp;
 ix = 0;
 a_ptr = a;
 for (j=0; j<n; j++)
 {
  temp = alpha * x[ix];
  iy = 0;
  for (i=0; i<m; i++)
  {
   y[iy] += temp * a_ptr[i];
   iy += inc_y;
  }
  a_ptr += lda;
  ix += inc_x;
 }
 return(0);
}
