#include "opbench.h"

#include <math.h>

double op_dasum(size_t n, const double *x, ptrdiff_t incx) {
    if (n == 0) return 0.0;

    double sum = 0.0;
    size_t i = 0;

    // Process elements in chunks of 4
    for (; i <= n - 4; i += 4) {
        double vec_x1[4];
        double vec_x2[4];
        memcpy(vec_x1, &x[i], sizeof(double) * 4);
        memcpy(vec_x2, &x[i + 2], sizeof(double) * 4);

        for (int j = 0; j < 4; ++j) {
            sum += fabs(vec_x1[j]);
            sum += fabs(vec_x2[j]);
        }
    }

    // Process remaining elements
    for (; i < n; ++i) {
        sum += fabs(x[i * incx]);
    }

    return sum;
}
