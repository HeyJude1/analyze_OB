#ifndef OPBENCH_H
#define OPBENCH_H

#include <stddef.h>

/* Vector pointers designate logical element zero, including negative strides. */
void op_daxpy(size_t n, double alpha, const double *x, ptrdiff_t incx,
              double *y, ptrdiff_t incy);
double op_ddot(size_t n, const double *x, ptrdiff_t incx,
               const double *y, ptrdiff_t incy);
void op_dscal(size_t n, double alpha, double *x, ptrdiff_t incx);
double op_dasum(size_t n, const double *x, ptrdiff_t incx);

/* Row-major, no transpose. Leading dimensions count elements. */
void op_dgemv(size_t m, size_t n, double alpha, const double *a, size_t lda,
              const double *x, ptrdiff_t incx, double beta, double *y, ptrdiff_t incy);
void op_dgemm(size_t m, size_t n, size_t k, double alpha, const double *a, size_t lda,
              const double *b, size_t ldb, double beta, double *c, size_t ldc);

/* The selected U/L triangle defines the whole row-major symmetric matrix. */
void op_dsymv(char uplo, size_t n, double alpha, const double *a, size_t lda,
              const double *x, ptrdiff_t incx, double beta, double *y, ptrdiff_t incy);
/* side is L when A multiplies B from the left, or R when A follows B. */
void op_dsymm(char side, char uplo, size_t m, size_t n, double alpha,
              const double *a, size_t lda, const double *b, size_t ldb,
              double beta, double *c, size_t ldc);

/* Valid linear convolution. The weights are reversed, contrary to correlation. */
void op_conv1d(size_t n, size_t k, const double *x, const double *w, double *y);

#endif
