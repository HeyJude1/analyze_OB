# 算子输入包

当前包含九个 FP64 任务。DAXPY、DDOT、DGEMV、DGEMM 是知识构建和生成联调的对象。DSCAL、DASUM、DSYMV 是 BLAS 留出，DSYMM 用于共享微内核背景下的近迁移评估，CONV1D 是滑窗计算的外部留出。

| 任务 | 约定 | AVX2 / RVV 变体覆盖的动作 |
| --- | --- | --- |
| DAXPY | `y ← αx + y`，支持正负非零步长 | 单位步长快速路径、FMA、固定展开与标量尾部 / 动态 `vl` |
| DDOT | 双向量点积，允许浮点归约重排 | 多累加器、水平归约 / 动态 `vl`、尾部保留和全 lane 归约 |
| DGEMV | 行主序、无转置，`y ← αAx + βy`，支持向量步长和行 padding | 沿连续行做点积，保留跨步回退 / 动态 `vl` 下的向量归约 |
| DGEMM | 行主序、无转置，`C ← αAB + βC`，传入标准未打包矩阵 | 沿输出列使用 FMA，寄存器暂存输出片段，尾部与 `β=0` 分支 |
| DSCAL | `x ← αx`，支持正负非零步长 | 单位步长向量乘及标量回退 |
| DASUM | 绝对值求和，支持正负非零步长 | AVX2 清符号位与双累加器，RVV 动态 `vl` 和向量归约 |
| DSYMV | 行主序、上下三角可选，`y ← αAx + βy` | 对存储三角的连续区间向量化，并对镜像区间读取正确索引 |
| DSYMM | 行主序、上下三角可选，支持 `A*B` 或 `B*A` | AVX2 向量化输出片段，RVV 左乘路径按动态 `vl` 向量化 |
| CONV1D | FP64 valid 一维卷积。`y[i] = Σ x[i+j]w[k-1-j]`，`k=0` 或 `k>n` 时无输出 | 跨输出位置载入连续窗口，保留每个位置的累加；AVX2 固定向量宽度，RVV 使用动态 `vl` |

[manifest.json](manifest.json) 是程序入口。每个任务有[公共头文件](include/opbench.h)中的独立符号、自己的 `tasks/<算子>/contract.json`、通用 C 输入和两种显式向量变体。契约包含数据布局、数值语义、查询特征和后续建议测量形状。向量指针指向逻辑第零项，负步长时应由调用者移动指针。矩阵的 leading dimension 以元素计，DGEMM 入口接收未打包矩阵。如果后续引入打包，端到端计时包含打包成本。

`generic.c` 是本项目编写的通用输入。编译器 `-O3` 基线应直接编译该文件并保留自动优化。`x86_avx2.c` 与 `riscv_rvv.c` 是同 ABI 的手工实现示例，用于验证生成和执行接口，也可用于同契约的人工候选参照。OpenBLAS 性能参照应通过目标平台实际构建的库和一致的外部调用接口获得。尤其不要直接把内部预打包微内核与接收标准矩阵的实现比较。

`vendor/openblas/` 保留 13 份 OpenBLAS 源文件副本，包括四份参考 BLAS 接口、x86 专家内核以及 RVV 专家内核和打包例程。文件来自本地 OpenBLAS 仓库的 `ce3f668c992cb3cc80849d5c30ed637f5adbd5b2` 提交。清单记录原始路径、内容哈希、接口层次和部分原文行号。它们为知识抽取提供观察依据，独立于本包的运行代码。许可证保留于 [OpenBLAS LICENSE](vendor/openblas/LICENSE) 及 [LAPACK LICENSE](vendor/openblas/lapack-netlib/LICENSE)。跨文件、实现层次或语言的对应关系应按优化行为解释，不能当作同 ABI 的逐行改动。

留出任务的通用 C 与向量版本按对应的数学定义独立实现。生成器可读通用 C 和接口契约。显式向量版本、功能驱动和详细评估输出保留在评测侧。DSYMM 按与 GEMM 共享部分专家内核的近迁移案例报告。留出只约束本项目的输入语料及派生知识，模型预训练可能接触相似实现。

## 复现编译和功能检查

进入本目录，使用 Python 3 和支持目标 ISA 的 GCC。x86 机器须支持 AVX2 与 FMA。

```powershell
python verify.py --target x86 --variant generic --report reports/x86-generic.json
python verify.py --target x86 --variant arch --report reports/x86-avx2.json
python verify.py --task dgemm --target x86 --candidate path/to/my_candidate.c
```

候选文件可包含 `opbench.h`，定义所选任务的符号即可。驱动与候选分别从独立源码编译。驱动检查允许契约内的边界尺寸、非整倍长、正负步长、矩阵 padding、`α=0`/`β=0`、输出边界和输入数据保留。当前检查使用固定随机种子和 `long double` 参考结果，作为公开冒烟用例。正式实验仍需独立数据、更多数值分布与形状。

DSCAL 覆盖四种系数和六种步长，DASUM 覆盖六种步长与三类数值分布。DSYMV、DSYMM 将未选三角区与输入填充区设为 NaN，使用另存的完整对称矩阵计算参考值，并检查输出越界和读入数组被改写的情况。DSYMM 遍历左右乘及上下三角组合。生成候选使用相同驱动。发生超时时，JSON 报告中的状态分别为 `compile_timeout` 或 `run_timeout`。

在 Linux/WSL 中使用 RISC-V 交叉编译器时，`--cc` 可以指定支持 RVV 1.0 intrinsic 的实际编译器路径。示例使用 QEMU user mode 和静态链接。

```bash
python3 verify.py --target rvv --variant arch --cc riscv64-linux-gnu-gcc \
  --cflag=-static --runner qemu-riscv64 --runner-arg=-cpu \
  --runner-arg=rv64,v=true,vlen=256,elen=64,vext_spec=v1.0 \
  --report reports/rvv-qemu256-arch.json
```

把 `vlen=256` 改成 `vlen=128` 可复核较短的向量寄存器配置。在 x86 主机上省略 `--runner` 会交叉编译并报告 `compiled_only`。RISC-V 真机支持 `RV64GCV` 与 FP64 且工具链匹配时，也可以 `--target rvv --cc gcc` 本地执行功能验证。

## 首批五算子验证记录

| 目标环境 | 版本 | 结果 | 原始报告 |
| --- | --- | --- | --- |
| 本机 x86，MinGW GCC 14.2.0 | 通用 C，`-O3` | 5 个任务，244/244 用例通过 | [x86 通用](reports/x86-generic.json) |
| 本机 x86，MinGW GCC 14.2.0 | AVX2/FMA 显式实现 | 5 个任务，244/244 用例通过 | [x86 AVX2](reports/x86-avx2.json) |
| GCC 14.3.0 交叉编译，QEMU 8.2.2，VLEN=256 | 通用 C，`-O3` | 5 个任务，244/244 用例通过 | [RVV 256 通用](reports/rvv-qemu256-generic.json) |
| GCC 14.3.0 交叉编译，QEMU 8.2.2，VLEN=256 | RVV 显式实现 | 5 个任务，244/244 用例通过 | [RVV 256 显式](reports/rvv-qemu256-arch.json) |
| GCC 14.3.0 交叉编译，QEMU 8.2.2，VLEN=128 | RVV 显式实现 | 5 个任务，244/244 用例通过 | [RVV 128 显式](reports/rvv-qemu128-arch.json) |

每组的用例组合为 DAXPY 60、DDOT 60、DGEMV 84、DGEMM 32、CONV1D 8。各组重复同一验证集，以上结果只支持编译、功能兼容和接口联调。后续真机实验需另行记录 CPU 或板卡、指令集、编译器、OpenBLAS 目标和实际计时样本。

## 九算子扩充后的验证记录

新增 DSCAL 384、DASUM 288、DSYMV 720、DSYMM 288 个用例，加上首批 244 个用例后，每组覆盖 1,924 个用例。五组均用 `-Werror` 编译，并通过全部功能用例。表中的结果来自项目实现的验证，模型生成结果见实验登记。

| 目标环境 | 通用 C | 显式向量实现 |
| --- | --- | --- |
| 本机 x86 | [1,924/1,924](reports/x86-generic-expanded.json) | [AVX2 1,924/1,924](reports/x86-avx2-expanded.json) |
| QEMU RVV 1.0，VLEN=256 | [1,924/1,924](reports/rvv-qemu256-generic-expanded.json) | [RVV 1,924/1,924](reports/rvv-qemu256-arch-expanded.json) |
| QEMU RVV 1.0，VLEN=128 | 沿用 VLEN=256 通用验证 | [RVV 1,924/1,924](reports/rvv-qemu128-arch-expanded.json) |

为复核检查器，将 DSYMM 的镜像读取故意改成直接读取，数值检查能稳定报告错误。另用挂起的 DASUM 候选验证了运行超时的 JSON 记录。报告中的 `failures` 计数为失败的断言，可大于用例数。
