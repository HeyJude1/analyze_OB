/* E09 修复回路自测夹具：编译失败（compile_failed 分支）
 * 故意使用不存在的 RVV intrinsic，且缺少头文件 → 交叉编译必然报错。
 * 用法（mock 生成用）：run_smoke.py --mock-code selftest/broken_compile_daxpy.c
 *   → 初始 compile_failed → repair 兜底返回 tasks/daxpy/input/generic.c → 重编译/QEMU 通过。
 */
typedef long long BLASLONG;

int daxpy_k(BLASLONG n, BLASLONG dummy0, BLASLONG dummy1, double da,
            double *x, BLASLONG inc_x, double *y, BLASLONG inc_y,
            double *dummy, BLASLONG dummy2)
{
    /* 幻觉 intrinsic：不存在 __riscv_vlseg1e64_v_f64m8，也没有 include riscv_vector.h */
    vfloat64m8_t vx = __riscv_vlseg1e64_v_f64m8(x, 1);
    (void)vx;
    (void)n; (void)da; (void)inc_x; (void)inc_y; (void)y;
    return 0;
}
