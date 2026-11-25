typedef long long BLASLONG;

int daxpy_k(BLASLONG n, BLASLONG dummy0, BLASLONG dummy1, double da, double *x, BLASLONG inc_x, double *y, BLASLONG inc_y, double *dummy, BLASLONG dummy2)
{
    BLASLONG i;
    BLASLONG ix = 0;
    BLASLONG iy = 0;

    if (n <= 0) return 0;
    if (da == 0.0) return 0;

    for (i = 0; i < n; i++) {
        y[iy] += da * x[ix];
        ix += inc_x;
        iy += inc_y;
    }
    return 0;
}
