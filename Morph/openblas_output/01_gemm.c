# 29 "../kernel/x86_64/../generic/gemm_small_matrix_kernel_nn.c" 2
int dgemm_small_kernel_b0_nn(BLASLONG M, BLASLONG N, BLASLONG K, double * A, BLASLONG lda, double alpha, double * B, BLASLONG ldb, double * C, BLASLONG ldc)
{
 BLASLONG i,j,k;
 double result=0.0;
 for(i=0; i<M; i++){
  for(j=0; j<N; j++){
   result=0.0;
   for(k=0; k<K; k++){
    result += A[i+k*lda] * B[k+j*ldb];
   }
   C[i+j*ldc]=alpha * result;
  }
 }
 return 0;
}
