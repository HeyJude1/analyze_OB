typedef long long BLASLONG;

#include <riscv_vector.h>
double ddot_k(BLASLONG n, double *x, BLASLONG inc_x, double *y, BLASLONG inc_y)
{
 BLASLONG i=0;
 BLASLONG ix=0,iy=0;

 if (n <= 0) return(0.0);

 if (inc_x == 1 && inc_y == 1) {
  size_t vlmax = __riscv_vsetvlmax_e64m8();
  vfloat64m8_t vsum = __riscv_vfmv_v_f_f64m8(0.0, vlmax);
  BLASLONG remaining = n;

  while(remaining >= vlmax) {
   size_t vl = __riscv_vsetvl_e64m8(vlmax);
   vfloat64m8_t vx = __riscv_vle64_v_f64m8(x, vl);
   vfloat64m8_t vy = __riscv_vle64_v_f64m8(y, vl);
   vsum = __riscv_vfmacc_vv_f64m8(vsum, vx, vy, vl);
   x += vl;
   y += vl;
   remaining -= vl;
  }

  if (remaining > 0) {
   size_t vl = __riscv_vsetvl_e64m8(remaining);
   vfloat64m8_t vx = __riscv_vle64_v_f64m8(x, vl);
   vfloat64m8_t vy = __riscv_vle64_v_f64m8(y, vl);
   vsum = __riscv_vfmacc_vv_f64m8(vsum, vx, vy, vl);
  }

  vfloat64m1_t vred = __riscv_vfredusum_vs_f64m8_f64m1(vsum, __riscv_vfmv_v_f_f64m1(0.0, 1), vlmax);
  return __riscv_vfmv_f_s_f64m1_f64(vred);
 } else {
  double dot = 0.0;
  while(i < n) {
   dot += y[iy] * x[ix];
   ix += inc_x;
   iy += inc_y;
   i++ ;
  }
  return(dot);
 }
}
