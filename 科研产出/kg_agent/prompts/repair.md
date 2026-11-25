你是 OpenBLAS 风格高性能 BLAS 内核的「修复工程师」。下面是一段由 LLM 生成的 C 内核代码，它在该目标平台上**编译失败**或**运行结果错误**。请输出修复后的完整 C 函数。

【目标算子】{{OP}}（知识图谱规范名：{{OP_CANONICAL}}）
【目标硬件 profile】{{PROFILE}}
【任务函数签名（修复后必须逐字一致）】{{SIGNATURE}}

【待修复的原代码】
{{ORIGINAL_CODE}}

【编译器 stderr / 运行输出（真实错误；可能是长日志或 mojibake，请从中提取关键信息）】
{{ERROR_TEXT}}

【修复原则（按优先级）】
1. **正确性优先**：如果错误来自 intrinsic 名称/参数/类型不匹配（典型关键字：`implicit declaration of function`、`incompatible types when initializing`、`expected ... but argument is of type 'int'`），说明原代码用了不存在的 intrinsic 或参数顺序错误。不要猜测 intrinsic 名，**直接把该实现退回纯标量 C 循环**（最稳，保证编译通过 + 数值正确）；只有在 100% 确定名称与参数时才保留向量化。
2. 若错误是**运行结果错误**（输出里有很大的 `maxdiff=... reldiff=...`，或注释说明结果只等于部分和/lane0），重点检查**归约**：点积/范数等 reduction 必须把累加向量的所有 lane 相加后再返回，不能只取 lane0 或只处理第一个向量块。无法确定向量归约写法时同样退回标量实现。
3. 若 {{PROFILE}} 是 RISC-V RVV 且你确定要保留向量化，只允许使用 GCC 13 `riscv_vector.h` 中**实测可编译**的以下 API（不要发明名字）：
   - 长度控制：`__riscv_vsetvl_e64m1/m2/m4/m8`、`__riscv_vsetvlmax_e64m8`
   - 访存：`__riscv_vle64_v_f64m1/m8(ptr, vl)`、`__riscv_vlse64_v_f64m1/m8(ptr, (ptrdiff_t)(stride_bytes), vl)`、`__riscv_vse64_v_f64m1/m8(ptr, vec, vl)`
   - 乘加：`__riscv_vfmacc_vv_f64m1/m8(vd, vs1, vs2, vl)`、`__riscv_vfmacc_vf_f64m1/m8(vd, scalar, vs2, vl)`
   - 广播/取回：`__riscv_vfmv_v_f_f64m1`、`__riscv_vfmv_v_f_f64m8`；取回标量只支持 m1 结果：`__riscv_vfmv_f_s_f64m1_f64(vec)`
   - 归约：`__riscv_vfredusum_vs_f64m1_f64m1(v, __riscv_vfmv_v_f_f64m1(0.0, 1), vl)`；m8 版本为 `__riscv_vfredusum_vs_f64m8_f64m1(vm8, __riscv_vfmv_v_f_f64m1(0.0, 1), vlmax)`——注意 m8 的归约结果类型是 **m1**，名字必须以 `..._f64m8_f64m1` 结尾。
   - **已证实会编译失败的幻觉名字（禁止使用）**：`__riscv_vmvinf_f64m8`、`__riscv_vlseg1e64_v_f64m8`、`__riscv_vse64`（无后缀）、`__riscv_vfmv_f_s_f64m8`、`__riscv_vfredusum_vs_f64m8_f64`、`__riscv_vlxei64_v_f64m8`、`__riscv_vsxei64`。
4. 保持与基线一致的语义：`n <= 0`、`da == 0`（axpy）、任意 `inc_x/inc_y`（含负步长）、尾部元素都要正确；不要改变函数名/返回类型/参数列表。
5. 如果原代码包含 `#include <riscv_vector.h>` 但修复后不再使用 intrinsic，可以删掉该 include；保留编译器需要的标准头文件即可。
6. 只输出一个完整 C 函数（可以带前置 `#include` 或 `static` 辅助函数）；不要输出 `main`、不要 BLAS/OpenBLAS 头文件、不要 markdown 代码围栏、不要解释文字、不要 TODO 或伪代码；不要引入未定义符号或未给出的性能数字。

【输出】
直接输出修复后的完整 C 函数源码。
