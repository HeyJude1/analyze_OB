typedef long long BLASLONG;

double ddot_k(BLASLONG n, double *x, BLASLONG inc_x, double *y, BLASLONG inc_y)
{
    if (n <= 0) return 0.0;

    double dot = 0.0;
    BLASLONG i;

    if (inc_x == 1 && inc_y == 1) {
        for (i = 0; i < n; i++) {
            dot += x[i] * y[i];
        }
    } else if (inc_y == 1) {
        for (i = 0; i < n; i++) {
            dot += x[i * inc_x] * y[i];
        }
    } else {
        for (i = 0; i < n; i++) {
            dot += x[i * inc_x] * y[i * inc_y];
        }
    }

    return dot;
}
