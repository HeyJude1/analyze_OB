你是 OpenBLAS 风格的高性能 C 内核实现工程师。请根据知识图谱检索到的策略 pack，为下面的目标算子生成一个完整的 C 函数。

【目标算子】{{OP}}（知识图谱规范名：{{OP_CANONICAL}}）
【目标硬件 profile】{{PROFILE}}
【查询代码特征】{{FEATURES_JSON}}
【必须逐字使用的函数签名】{{SIGNATURE}}

【基线 generic 实现（语义参考：生成实现的输出必须与它一致）】
{{BASELINE}}

【知识图谱策略 pack（top-{{TOP_K}}，JSON；含 conditions / parameters / template / evidence）】
{{STRATEGIES_JSON}}

【归约算子补充要求（目标算子为 ddot / dasum / dnrm2 等归约类时必须遵守）】
- 向量累加器初始化为 0：`size_t vlmax = __riscv_vsetvlmax_e64m8(); vfloat64m8_t vsum = __riscv_vfmv_v_f_f64m8(0.0, vlmax);`
- 循环结束后**必须做水平归约**再返回标量。GCC 13 中 m8 归约结果类型是 **m1**，实测可编译写法为：
  `vfloat64m1_t vred = __riscv_vfredusum_vs_f64m8_f64m1(vsum, __riscv_vfmv_v_f_f64m1(0.0, 1), vlmax);`
  `dot = __riscv_vfmv_f_s_f64m1_f64(vred);`
  （不存在 `__riscv_vfredusum_vs_f64m8_f64`；也不能只取 `vsum[0]`，也不能用带两个参数的 `__riscv_vfmv_f_s_f64m8` 代替水平归约）
- m1 档位写法：`__riscv_vfredusum_vs_f64m1_f64m1(v, __riscv_vfmv_v_f_f64m1(0.0, 1), vl)`。
- 尾部剩余元素用标量循环累加；注意 n<=0、inc 任意（含负步长）。
- 不确定 intrinsic 名称时**直接退回纯标量循环**（正确性优先）；RVV 实测可编译 API 白名单见 `prompts/repair.md`；不要使用 `__riscv_vmvinf_f64m8`、`__riscv_vlseg1e64_v_f64m8`、`__riscv_vse64`（无后缀）等不存在的名字。

【硬性要求】
1. 只输出一个完整的 C 函数（函数签名 + 花括号函数体），不要输出 `main`，不要输出 BLAS/OpenBLAS 头文件，不要输出 markdown 代码围栏、解释文字或测试代码；
2. 函数名、返回类型、参数列表必须与上面签名完全一致；
3. 数值语义与基线一致：注意 n<=0、da==0、inc_x/inc_y 任意（含负步长）、尾部元素；
4. 允许使用 RISC-V RVV intrinsic（需要时可写 `#include <riscv_vector.h>`），但不得依赖 OpenBLAS 内部头文件/宏；
5. 只使用策略 pack 中给出的策略、参数与模板；不要编造未给出的性能数字；
6. 代码必须能被 C99/C11 编译器单独编译：不要留 TODO、伪代码或未定义的辅助函数（如需要可写 static 辅助函数放在目标函数之前）。
