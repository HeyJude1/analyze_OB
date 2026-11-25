typedef long long BLASLONG;

double ddot_k(BLASLONG n, double *x, BLASLONG inc_x, double *y, BLASLONG inc_y)
{
    BLASLONG i;
    BLASLONG ix = 0;
    BLASLONG iy = 0;
    double acc = 0.0;

    for (i = 0; i < n; i++) {
        acc += x[ix] * y[iy];
        ix += inc_x;
        iy += inc_y;
    }
    return acc;
}
