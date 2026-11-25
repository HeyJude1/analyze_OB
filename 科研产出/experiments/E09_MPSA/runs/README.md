# runs 目录说明

- 每个 run 只有一个 `.md` 文件：`YYYY-MM-DD_HHMM__<run_id>__<状态>.md`。
- `INDEX.md` 是总索引；正式 run 在根目录，mock/selftest 在 `_mock/`、`_selftest/`。
- 原始产物在 `../artifacts/<run_id>/`；可复现包在 `../artifacts/<run_id>/repro/README.md`（输入/硬件/检索链/每轮迭代/最终算子）。
- 报告内链接已自动改写为 `../artifacts/...`。
- 状态：OK / MIXED / FAIL / MOCK / SELFTEST。
