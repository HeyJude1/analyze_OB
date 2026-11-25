/* OpenBLAS expert adapter for the dense, nondegenerate timing workloads.
 * Build only with the installed OpenBLAS CBLAS header and matching library.
 */
#include "opbench.h"
#include <cblas.h>

#if defined(OPBENCH_TASK_DAXPY)
void op_daxpy(size_t n, double alpha, const double *x, ptrdiff_t incx,
              double *y, ptrdiff_t incy) {
    cblas_daxpy(n, alpha, x, incx, y, incy);
}
#elif defined(OPBENCH_TASK_DDOT)
double op_ddot(size_t n, const double *x, ptrdiff_t incx,
               const double *y, ptrdiff_t incy) {
    return cblas_ddot(n, x, incx, y, incy);
}
#elif defined(OPBENCH_TASK_DGEMV)
void op_dgemv(size_t m, size_t n, double alpha, const double *a, size_t lda,
              const double *x, ptrdiff_t incx, double beta, double *y, ptrdiff_t incy) {
    cblas_dgemv(CblasRowMajor, CblasNoTrans, m, n, alpha, a, lda, x, incx, beta, y, incy);
}
#elif defined(OPBENCH_TASK_DGEMM)
void op_dgemm(size_t m, size_t n, size_t k, double alpha, const double *a, size_t lda,
              const double *b, size_t ldb, double beta, double *c, size_t ldc) {
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}
#else
#error Choose a BLAS task with OPBENCH_TASK_<OP>
#endif
