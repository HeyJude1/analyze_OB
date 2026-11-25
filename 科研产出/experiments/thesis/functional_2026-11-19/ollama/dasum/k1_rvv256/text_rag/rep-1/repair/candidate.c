#include "opbench.h"

#include <math.h>

double op_dasum(size_t n, const double *x, ptrdiff_t incx) {
    double sum = 0.0;
    for (size_t i = 0; i < n; ++i) sum += fabs(x[(ptrdiff_t)i * incx]);
    return sum;
}
