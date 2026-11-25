# 论文实验的准备工具

[九算子首轮功能对照](functional_2026-11-19/README.md)保存本机与 QEMU 的四组实际生成和修复记录。批量执行由 `run_functional.py` 完成，`summarize_functional.py` 按平台、方法和划分读取全量计划并汇总首轮与末轮结果。

从 `科研产出` 目录运行。每次使用新的输出目录，可固定算子和平台缩小调试范围。

```powershell
python experiments/thesis/prepare_comparison.py --out experiments/thesis/prepared/pilot1
```

早期五算子输入见[120 单元清单](artifacts/prepared-v1/manifest.json)。当前九算子可通过上述命令从图谱和算子输入包生成新的对照输入。

输出目录包含实验单元清单、基准输入和图谱快照的哈希、提示词、策略卡 ID、字节占用及两份检索轨迹。同一个算子与平台共享基础提示、接口与通用代码。所有知识组使用同一个图谱的适用卡片，文本组按 BM25 排序，策略卡组独立排序，规划组附加关系和目标参数候选。程序选择能放进共同 UTF-8 字节上限的完整卡片，匹配三组卡片数。正式调用需按所用模型核对 token 预算和各调用阶段的成本。

名称中的 `x86_avx2` 与 `k1_rvv256` 表示初步静态查询配置。板卡型号、编译选项和 ISA 能力应在真机取样后冻结。卷积以 DOT 作为人工声明的类比起点，此项设置须与实验结果一起披露。

准备工具的 `prepared` 状态表示输入生成完成。模型生成、修复、候选验证与真机测量的进度另记，未执行的单元从结果汇总中明确标出。

以下命令将该单元的通用输入回放为候选文件，可离线验证清单、提示哈希及编译接口。回放状态为 `replayed_input`。为正式调用显式指定 `--engine dashscope --model <固定模型版本>`，并通过环境设置 API 密钥。调用返回的 `generated_unverified` 还需单独编译、校验、计时。重复序号用于配对独立调用，当前客户端不向服务端传 seed。

```powershell
python experiments/thesis/generate_candidate.py --prepared experiments/thesis/prepared/pilot1 --task dgemm --profile x86_avx2 --method constraint_plan --out experiments/thesis/.build/replay-dgemm
python 算子输入/verify.py --task dgemm --target x86 --candidate experiments/thesis/.build/replay-dgemm/candidate.c
```

## 成对计时

候选通过独立正确性检查后，可使用 `build_measure.py` 将它与同题通用 C 编译进同一个 Linux 可执行文件。工具只编译，保留命令、日志、源码和运行文件的哈希。下面是 DGEMM 的原生 x86 示例。RISC-V 可改用已核实的本地或交叉编译器，并选用与实测板卡匹配的 ISA 参数。

```bash
python3 experiments/thesis/build_measure.py --task dgemm \
  --candidate 算子输入/tasks/dgemm/x86_avx2.c \
  --cflag=-mavx2 --cflag=-mfma --out experiments/thesis/build/dgemm-demo
# 在安排真机测量后执行，最后两项分别是每批调用次数和基线/候选配对数。
taskset -c 2 experiments/thesis/build/dgemm-demo/measure 96 103 127 30 25 > dgemm-raw.csv
```

计时驱动交替测量顺序，输出每对的原始批时长。首版使用连续、对齐且非退化的输入，DAXPY 在每批计时前清空输出，矩阵接口使用完整调用。分别试跑以校准批长度、热状态和核编号。非连续与退化样本由正确性框架覆盖，测它们的性能须使用相应工作负载驱动。

OpenBLAS 参照使用 `--candidate experiments/thesis/openblas_candidate.c --ldflag=-lopenblas`。按本机编译器配置添加 OpenBLAS 的头文件和库目录，选择与板卡架构匹配的库。计时前设置 `OPENBLAS_NUM_THREADS=1` 和 `OMP_NUM_THREADS=1`，记录库的版本、目标和链接方式。包装只覆盖上述连续、非退化的计时场景。它以完整 CBLAS 调用提供专家参照，并包含库内布局转换、打包和分派成本。

五个自带算子在 Windows x86 及 RV64GCV Linux 目标下均已构建成对计时程序。四个 BLAS 的 OpenBLAS 包装已用本地 RVV 静态库交叉链接，示例见 [DGEMM 的构建记录](artifacts/rvv-openblas-dgemm/build.json)。这份库的配置要求 VLEN 至少 256 位，部署前核对真机及实际调度目标。x86 OpenBLAS 链接和各程序的真机计时待执行。
