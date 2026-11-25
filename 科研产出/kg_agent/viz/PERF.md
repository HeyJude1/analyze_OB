# 可视化性能优化记录（PERF）

> 2026-11-12｜kg-architect｜对应任务 task-12
> 用户反馈：「目前的可视化界面太卡了，有没有性能好点的项目」
> 结论：**两条路径都已落地** —— ① 现有 `graph.html` 就地改成"性能模式 + 懒加载 + 增量刷新"（首屏约 **306ms**，旧版 **1684ms**）；② 新增 `graph_fast.html`（Sigma.js **WebGL**，默认算子 2 跳子图 60 节点，全图 862/3656 可拖拽缩放，首屏约 **544ms**）。

> ⚠️ 数据版本说明：任务书写的是 527 节点 / 2218 边，执行期间 pipeline 又更新了数据，**本文所有实测基于当前 `graph_data.js`：862 节点 / 3656 边 / 6 条 trace**。页面内所有统计均为运行时计算，不依赖固定数字。

---

## 1. 改了什么

### A. `graph.html`（原地优化，仍为默认入口）

| # | 优化 | 说明 |
|---|---|---|
| A1 | **性能模式默认开启** | 只渲染 `Operator / ComputationPattern / OptimizationStrategy / ImplementationTemplate` 四类核心实体：可见 **614 / 862** 节点、**921 / 3656** 边；其余类型在类型面板标「折叠」，按需加载 |
| A2 | **关物理引擎** | 用一次确定性的轻量力导向布局（采样斥力 FR，862 节点约 **25ms**）算好坐标，`physics.enabled=false`、不再 `stabilize()`（旧版 stabilize 260 轮是首屏慢的主因） |
| A3 | **标签按需显示** | 默认不渲染标签；仅在 **选中 / hover / 缩放 > 1.3** 时显示（缩放阈值穿越时才批量更新一次） |
| A4 | **增量更新** | 节点/边样式做 diff 缓存，选中切换只更新前后 2 个节点（旧版每次交互全量更新 862 节点 + 3656 边，约 **122ms**） |
| A5 | **拖拽优化** | `hideEdgesOnDrag` / `hideNodesOnDrag` 开启，拖拽时隐藏边与节点；边 `smooth=false`（直线），显著降低 canvas 绘制量 |
| A6 | **懒加载交互** | 点击节点 → 详情面板「**展开 1 跳邻居**」（自动加载其折叠邻居并聚焦）；「显示全图」开关一键切到全量 862 节点，可随时回到性能模式 |

其余 5 项交互（图探索 / 路径 / 推理链 / 检索回放 / 搜索+图例）全部保留，`?selftest=1` **18/18 通过**。

### B. `graph_fast.html`（新增，WebGL 性能版）

- **Sigma.js（WebGL，本地 `fast_assets/sigma.min.js`）+ Graphology**，本地 `fast_assets/graphology.umd.min.js`，不联网。
- **默认只渲染当前算子的 2 跳子图**（gemm → 60 节点 / 254 边，含算子下拉可切换 12 个算子）；「加载全图」按钮切换到 **862 节点 / 3656 边**，再点「回到子图」返回。
- 标签：子图模式默认显示（sigma label grid 防重叠）；全图模式只在 hover / 选中显示。
- `nodeReducer` / `edgeReducer` 走**无分配快路径**（无高亮时直接返回原 data），高亮/变暗/隐藏全在 reducer 里完成，**不重建图**。
- 交互：类型过滤、搜索（防抖 + 高亮）、点击详情（属性 + 来源行号 + 关联证据 + 展开邻居）、BFS 路径（自动加载子图外路径节点 + 相机跟随）、7 步检索回放（上一步/下一步/自动播放/重置 + 候选/选中/淘汰 + reason/dropped）。
- `?selftest=1` **12/12 通过**；`?full=1`、`?op=<nodeId>`、`?panel=replay&step=4` 等深链接可用。

---

## 2. 实测对比（同一数据 862 节点 / 3656 边）

测量方法：Chrome 157 headless + **CDP（远程调试）** 在页面内用 `performance.now()` 实测，每项 2–3 次取范围；`file://` 直接打开，**无服务器**。旧版 = 优化前 `graph.html` 的快照副本（同一份数据、同一测量方式）。

| 指标 | 旧版 graph.html（Canvas 全量） | 新版 graph.html（性能模式） | graph_fast.html（Sigma WebGL） |
|---|---|---|---|
| 首屏（到可交互/`__KG_VIZ__` 就绪） | **1648 ~ 1720 ms** | **272 ~ 339 ms**（≈5.5× 更快） | **537 ~ 551 ms**（首帧 555~571 ms） |
| 默认渲染量 | 862 节点 / 3656 边 | **614 节点 / 921 边** | **60 节点 / 254 边**（gemm 2 跳） |
| 布局 | vis 物理 stabilize 260 轮 | 预计算 **24~29 ms** | 预计算 **22~27 ms** |
| 全量状态刷新（搜索/回放/路径切换） | **≈122 ms**（每次交互都触发） | **83~96 ms**（仅镜头切换时触发；选中走增量） | **≈2.9 ms/帧**（CPU 提交，GPU 绘制异步） |
| 点击选中 | ≈122 ms（全量更新） | **≈22.9 ms**（含详情面板 DOM 渲染） | **≈9.4 ms**（含详情渲染） |
| hover 标签 | 全量重绘 | **≈10.5 ms**（单节点 DataSet 更新） | <1 ms（reducer + refresh） |
| 加载全图 / 切换 | 默认就是全图 | **≈47 ms/次**（显示全图↔性能模式） | **3.1~3.4 ms**（862/3656 即时切换） |
| 全图选中切换 | ≈122 ms | 47 ms（切换模式时） | **2.6~3.0 ms** |
| canvas 重绘（单帧） | **22.1~24.2 ms** | **3.7 ms**（关边曲线/标签/物理后） | WebGL（无 canvas 逐节点绘制） |
| 5 项交互自检 | — | **18/18** | **12/12（含全图）** |

**为什么快**：旧版瓶颈不在"节点多"，而在 ① 物理引擎 stabilize；② 每次 hover/选中/高亮都全量 `DataSet.update`（862+3656 项，≈122ms）；③ 862 个标签 + 3656 条曲线边每次都重绘。新版分别用"预计算布局 + 增量 diff + 标签按需 + 直线边 + 隐藏拖拽"解决；WebGL 版则把绘制交给 GPU。

---

## 3. 怎么用 / 怎么切换

| 场景 | 打开 | 说明 |
|---|---|---|
| 日常使用（推荐） | 双击 `graph.html` | 默认性能模式（614/862）；点节点看详情、按「展开 1 跳邻居」加载周边，或勾选「显示全部节点」看全图 |
| 数据量大 / 想要最流畅 | 双击 `graph_fast.html` | 默认 gemm 2 跳（60 节点）；顶部选算子切换子图；「加载全图」看 862/3656 |
| 只想看全图 | `graph.html` → 顶部「显示全图」；或 `graph_fast.html` → 「加载全图」；或 URL `graph_fast.html?full=1` |
| 指定算子子图 | `graph_fast.html?op=op:gemm` | 12 个算子均可 |
| 回放某个 trace 第 4 步 | `graph.html?panel=replay&step=4` / `graph_fast.html?panel=replay&step=4` | 深链接同时支持 `select`、`from/to`、`q`、`trace` |
| 自检 | `graph.html?selftest=1`（18/18）、`graph_fast.html?selftest=1`（12/12） | 结果写入 `body[data-selftest]` 与 `#selftest-log` |
| 性能基准 | `graph.html?bench=1`、`graph_fast.html?full=1&bench=1` | 结果写入 `window.__KG_BENCH__` / `__KG_FAST_BENCH__` 并打印 console |

旧版对比截图：`_screenshots/old_full_vis.png`（862 节点全量 vis，首屏 1.6s+）；新版：`_screenshots/01_graph.png`（性能模式）、`_screenshots/fast_subgraph.png`、`_screenshots/fast_full.png`。

---

## 4. 已知限制（诚实说明）

1. **headless 环境下的 FPS 数字不可信**：Chrome headless 用 SwiftShader 软件光栅化 WebGL，rAF 实测只有 1–2 fps，不能代表真实 GPU 浏览器；本文以"首屏耗时 / 单帧 CPU 提交耗时 / 渲染量"作为可复现指标。**真实浏览器（有 GPU）的拖拽帧率需要用户在本机人工确认**。
2. **旧版基线是快照副本**：为了对比，旧版页面复制到临时目录并加了一个计时探针（不改业务逻辑），测量方式与新版本一致。
3. **数据仍在被 pipeline 更新**：本文数字对应 862/3656；旧任务书写的 527/2218 是上一版数据。重新 `export-viz` 后统计会变，但优化机制不变。
4. `graph_fast.html` 的 UI 是精简版（详情 / 路径 / 回放三个面板，没有"推理链"卡片；推理链请用 `graph.html`）。
5. WebGL 依赖浏览器硬件加速：极老机器/远程桌面可能创建 WebGL 上下文失败，页面会给出提示并建议使用 `graph.html`（Canvas 版）。
6. 性能模式下非核心类型默认折叠：搜索/路径/回放命中折叠节点时会自动展开，但"图探索"需要用户点「展开 1 跳邻居」才会看到周边证据/来源节点 —— 这是有意的性能取舍。

---

## 5. 结论

- **日常入口保持 `graph.html`**，默认性能模式已把首屏从 ~1.68s 降到 ~0.31s、单次交互从 ~122ms 降到 ~10–23ms；
- **要展示"全图 862 节点"时用 `graph_fast.html`**，WebGL 下全图切换 ~3ms、单帧 CPU 提交 ~2.9ms，是"性能好点的方案"的直接答案；
- 两版共享同一份 `graph_data.js` / `traces.js` 与同一套实体/关系语义，可随时对照。
