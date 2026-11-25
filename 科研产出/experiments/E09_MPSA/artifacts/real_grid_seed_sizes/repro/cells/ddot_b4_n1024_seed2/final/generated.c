typedef long long BLASLONG;

double ddot_k(BLASLONG n, double *x, BLASLONG inc_x, double *y, BLASLONG inc_y)
{
    if (n <= 0) return 0.0;

    double dot = 0.0;

    if (inc_x == 1 && inc_y == 1) {
        BLASLONG i = 0;
        for (; i < n; ++i) {
            dot += x[i] * y[i];
        }
    } else {
        BLASLONG i = 0;
        for (; i < n; ++i) {
            dot += x[i * inc_x] * y[i * inc_y];
        }
    }

    return dot;
}
