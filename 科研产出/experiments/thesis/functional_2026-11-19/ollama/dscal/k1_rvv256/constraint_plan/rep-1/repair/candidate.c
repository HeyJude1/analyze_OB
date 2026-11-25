#include "opbench.h"

void op_dscal(size_t n, double alpha, double *x, ptrdiff_t incx) {
    if (n == 0 || alpha == 1.0) return;

    size_t i = 0;
    for (; i < n - 3; i += 4) {
        x[i * incx] *= alpha;
        x[(i + 1) * incx] *= alpha;
        x[(i + 2) * incx] *= alpha;
        x[(i + 3) * incx] *= alpha;
    }

    for (; i < n; ++i) {
        x[i * incx] *= alpha;
    }
}
