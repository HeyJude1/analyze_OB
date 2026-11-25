typedef long long BLASLONG;

int daxpy_k(BLASLONG n, BLASLONG dummy0, BLASLONG dummy1, double da, double *x, BLASLONG inc_x, double *y, BLASLONG inc_y, double *dummy, BLASLONG dummy2)
{
    if (n <= 0) return 0;
    if (da == 0.0) return 0;

    if (inc_x == 1 && inc_y == 1) {
        for (BLASLONG i = 0; i < n; i++) {
            y[i] += da * x[i];
        }
    } else if (inc_y == 1) {
        BLASLONG ix = 0;
        for (BLASLONG i = 0; i < n; i++) {
            y[i] += da * x[ix];
            ix += inc_x;
        }
    } else {
        BLASLONG ix = 0, iy = 0;
        for (BLASLONG i = 0; i < n; i++) {
            y[iy] += da * x[ix];
            ix += inc_x;
            iy += inc_y;
        }
    }

    return 0;
}
