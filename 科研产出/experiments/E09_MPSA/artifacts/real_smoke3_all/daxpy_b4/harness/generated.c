typedef long long BLASLONG;

#include <riscv_vector.h>
int daxpy_k(BLASLONG n, BLASLONG dummy0, BLASLONG dummy1, double da, double *x, BLASLONG inc_x, double *y, BLASLONG inc_y, double *dummy, BLASLONG dummy2)
{
    if (n <= 0) return 0;
    if (da == 0.0) return 0;

    BLASLONG i = 0;
    const size_t vlmax = __riscv_vsetvlmax_e64m8();
    vfloat64m8_t vacc;

    // Fast path for unit increments
    if (inc_x == 1 && inc_y == 1) {
        for (; i <= n - vlmax; i += vlmax) {
            size_t vl = __riscv_vsetvl_e64m8(n - i);
            vacc = __riscv_vfmacc_vf_f64m8(__riscv_vle64_v_f64m8(y + i, vl), da, __riscv_vle64_v_f64m8(x + i, vl), vl);
            __riscv_vse64(y + i, vacc, vl);
        }
    } else {
        BLASLONG ix = 0;
        BLASLONG iy = 0;
        for (; i <= n - vlmax; i += vlmax) {
            size_t vl = __riscv_vsetvl_e64m8(n - i);
            vacc = __riscv_vfmacc_vf_f64m8(__riscv_vlxei64_v_f64m8(y, iy, vl), da, __riscv_vlxei64_v_f64m8(x, ix, vl), vl);
            __riscv_vsxei64(y, iy, vacc, vl);
            ix += inc_x * vlmax;
            iy += inc_y * vlmax;
        }
    }

    // Scalar fallback for remaining elements
    for (; i < n; i++) {
        y[i * inc_y] += da * x[i * inc_x];
    }

    return 0;
}
