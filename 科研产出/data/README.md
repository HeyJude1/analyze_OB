# data 目录说明

本目录是 Lead 对用户既有 RISC-V benchmark 日志的解析结果（2026-11-12 生成）。

| 文件 | 内容 |
|---|---|
| `riscv_log_summary.csv` | 44 个算子的汇总：通过率、`Migration/Ref` 几何平均（过滤 size<32）、中位数、Ref 时间中位数、日志自报的 avg/max/min、runner 判定 |
| `riscv_log_all_sizes.csv` | 所有 (算子, 尺寸) 明细：ref_ms、mig_ms、speedup、pass |

生成脚本：`科研产出/tools/parse_riscv_logs.py`
数据来源：`C:\Users\MSI\Desktop\cx_operator_tool\operator_extract\openblas_newtest1\<op>_op\*_log.txt`

## 关键结论（务必先读）

1. **44/44 算子全部 pass**，说明"提取出的 kernel"在功能上是正确的，可作为正确性回归基线。
2. `Migration/Ref` 跨算子几何平均中位数 ≈ **1.01**；最好 dgemv 2.57 / dsymm 2.45 / dgemm 2.20；最差 idmin/idmax/idamax/idamin ≈ 0.18–0.20、zdotu 0.52、dnrm2 0.56。
3. **这批日志的运行命令包含 `qemu-riscv64`（44/44），不是真机性能数据**；只能用于正确性验证与格式参考。真机实验必须重跑。
4. 那 4 个 0.18× 的算子是很好的"反面案例"：说明"从库里抽出来的实现"并不天然更快，正好论证"为什么 KG 要记录性能证据，而不是假设专家实现一定更快"。
