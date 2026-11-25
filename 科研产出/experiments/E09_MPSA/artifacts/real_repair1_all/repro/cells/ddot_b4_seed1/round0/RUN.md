# ddot 复现

```bash
cd C:/Users/MSI/Desktop/开题/科研产出/experiments/E09_MPSA/artifacts/real_repair1_all/repro/cells/ddot_b4_seed1/round0
bash build.sh && bash run.sh
```

- 期望：`maxdiff` 在容差内（见 `result.json` / `task.yaml`）
- QEMU 计时只用于确认能跑通，不是性能结论。
