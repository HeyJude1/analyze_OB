# ddot_b4_n16384 seed3

- 输入：`../../inputs/ddot/generic.c`；硬件：`../../inputs/hardware/hardware_profiles.yaml`
- 检索链：`../../inputs/retrieval/ddot_k1_rvv256_k6_trace.json`（pack 同目录）

## 迭代
- `round0/`：初始生成
- `final/`：最终代码 + `bin`，可直接 `bash build.sh && bash run.sh`

## 复现
```bash
cd final && bash build.sh && bash run.sh
```
