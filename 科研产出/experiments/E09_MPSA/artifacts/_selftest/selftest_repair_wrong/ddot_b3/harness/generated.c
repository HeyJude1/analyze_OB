/* E09 修复回路自测夹具：数值错误（wrong_answer 分支）
 * 编译通过、但恒返回 0.0 → driver 的 maxdiff≈8192 / reldiff≈1 → exit 2。
 * 用法（mock 生成用）：run_smoke.py --mock-code selftest/wrong_answer_ddot.c
 *   → 初始 wrong_answer（--error 传 qemu.stdout.log）→ repair 兜底返回 tasks/ddot/input/generic.c → 重跑通过。
 */
typedef long long BLASLONG;
double ddot_k(BLASLONG n, double *x, BLASLONG inc_x, double *y, BLASLONG inc_y)
{
    (void)n; (void)x; (void)inc_x; (void)y; (void)inc_y;
    return 0.0;
}
