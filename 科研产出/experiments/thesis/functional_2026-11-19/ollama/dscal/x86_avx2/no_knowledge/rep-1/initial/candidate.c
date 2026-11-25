#include "opbench.h"

void op_dscal(size_t n, double alpha, double *x, ptrdiff_t incx) {
    if (incx == 1) {
        for (size_t i = 0; i < n; ++i) x[i] *= alpha;
    } else if (incx > 0) {
        for (size_t i = 0; i < n; ++i) x[(ptrdiff_t)i * incx] *= alpha;
    } else {
        for (size_t i = 0; i < n; ++i) x[(-i - 1) * incx] *= alpha;
    }
}
