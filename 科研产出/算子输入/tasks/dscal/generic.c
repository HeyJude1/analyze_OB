#include "opbench.h"

void op_dscal(size_t n, double alpha, double *x, ptrdiff_t incx) {
    for (size_t i = 0; i < n; ++i) x[(ptrdiff_t)i * incx] *= alpha;
}
