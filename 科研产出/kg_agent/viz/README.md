# kg_agent 可视化（viz）

两个自包含单页，**双击用 `file://` 直接打开即可**，不联网、中文正常，数据接口严格按 `../SPEC.md` §2：

| 页面 | 定位 | 首屏实测 | 默认渲染量 |
|---|---|---|---|
| **`graph.html`** | 默认入口（vis-network Canvas，**性能模式 + 懒加载**） | ≈306 ms（旧版 ≈1684 ms） | 614 / 862 节点、921 / 3656 边 |
| **`graph_fast.html`** | WebGL 性能版（**Sigma.js + Graphology**） | ≈544 ms（首帧 ≈563 ms） | 算子 2 跳子图 60 / 254，一键全图 862 / 3656 |

> 数据：`viz/graph_data.js`（`window.KG_DATA`）+ `viz/traces.js`（`window.KG_TRACES`），由 pipeline 的 `export-viz` 生成；
> 当前真实数据为 **862 节点 / 3656 边 / 6 条 trace**。文件缺失时自动降级 `sample/`（32 节点 / 44 边 / 2 条 trace），
> 因此没有 pipeline 也能开发与演示。性能数据与测量方法见 **[PERF.md](PERF.md)**。

---

## 1. 快速开始

| 方式 | 操作 |
|---|---|
| 日常使用 | 双击 `graph.html`（Chrome / Edge），地址栏是 `file:///...` 也算正常 |
| 要大图/最流畅 | 双击 `graph_fast.html`，顶部选算子看 2 跳子图；点「加载全图」看 862/3656 |
| 自检 | `graph.html?selftest=1` → **21/21**；`graph_fast.html?selftest=1` → **15/15**（结果写入 `body[data-selftest]` 与 `#selftest-log`） |
| 性能基准 | `graph.html?bench=1`；`graph_fast.html?full=1&bench=1`（结果写入 `window.__KG_BENCH__` / `window.__KG_FAST_BENCH__` 并打印 console） |
| 演示某视图 | `graph.html?panel=replay&step=4`、`graph.html?panel=chain&select=<id>`、`graph_fast.html?full=1` 等（见 §6） |

> 不需要服务器。若浏览器策略禁止本地脚本，可 `python -m http.server 8000` 后访问
> `http://localhost:8000/graph.html`（正常双击即可，无需此步）。

## 2. 文件结构

```text
viz/
├── graph.html                    # 默认入口：图探索/路径/推理链/检索回放/搜索（性能模式，21 项自检）
├── graph_fast.html               # WebGL 性能版：Sigma.js，默认算子 2 跳子图 + 加载全图（15 项自检）
├── assets/vis-network.min.js     # vis-network 9.1.9（本地）
├── fast_assets/
│   ├── graphology.umd.min.js     # Graphology（本地）
│   └── sigma.min.js              # Sigma.js v3 WebGL（本地）
├── graph_data.js                 # ← pipeline export-viz 生成（缺失自动用 sample）
├── traces.js                     # ← pipeline 生成（缺失自动用 sample）
├── sample/
│   ├── sample_graph_data.js      # 示例图谱：32 节点 / 44 边 / 11 类实体
│   └── sample_traces.js          # 示例 trace：2 条 7 步检索过程
├── PERF.md                       # 性能优化记录：前后对比、测量方法、限制
├── _screenshots/                 # 验收截图（见 §8）
└── README.md
```

## 3. `graph.html`（默认入口）五项交互

> **性能模式（默认开启）**：只渲染 `Operator / ComputationPattern / OptimizationStrategy / ImplementationTemplate`
> 四类核心实体（当前 614/862 节点）；其余类型在左侧标「折叠」，点节点详情里的 **「展开 1 跳邻居」** 按需加载；
> 顶部 **「显示全图」** 一键切到 862 节点 / 3656 边，再点「回到性能模式」恢复。
> 标签仅在 **选中 / hover / 缩放 > 1.3** 时显示；物理引擎已关闭（预计算布局），拖拽时隐藏边/节点。

| # | 功能 | 操作 |
|---|---|---|
| ① | **图探索 / 点击聚焦** | 左侧类型过滤折叠/展开；节点大小 = 度数；**单击节点 = 聚焦**：该节点 + 全部 1 跳邻居高亮、其余节点/边降透明（0.12）、非相关标签隐藏，状态栏提示、详情面板按关系类型分组列出「关联节点」；**点画布空白 / Esc** 退出聚焦；详情页可「展开 1 跳邻居」 |
| ② | **路径跳转** | 填起点/终点（下拉或输入名称/ID）→「查找最短路」；绿色高亮路径 + 关系列表（含 `support`、`role` 等边属性）；「⇄」交换、「清除」取消 |
| ③ | **推理链** | 点击「优化策略」节点 → 右侧 5 张链式卡片：**适用条件 → 硬件约束 → 参数推导 → 性能证据 → 来源**；卡片内条目可点击跳转 |
| ④ | **检索回放** | 选算子 / 硬件 profile / trace → 7 步轨道（流程匹配→策略召回→条件过滤→硬件过滤→冲突消解→打分选择→组装）；`◀`/`▶` 步进、`⏸ 播放`、`⟲ 重置`；图中 **橙框=候选、绿框=选中、红框=淘汰**，右侧显示 `description / reason / dropped 原因 / 打分表` |
| ⑤ | **搜索 + 图例** | 搜索框按名称/类型/ID 过滤（回车选第一个，Esc 清空），命中紫框、其余变暗；图例含实体颜色、节点大小含义、关系线型（绿虚线=组合、红虚线=冲突） |

### 3.1 聚焦 / 重置 / Esc 速查

| 操作 | 效果 |
|---|---|
| **单击节点** | **聚焦**：选中节点 + 全部 1 跳邻居（所有关系类型）高亮、中心节点放大加粗；其余节点/边降透明到 0.12、非相关标签隐藏；状态栏显示「已聚焦：xxx（N 个邻居）· 点空白/Esc/重置退出」 |
| 点详情面板「关联节点（按关系）」里的邻居 | 聚焦切换到该邻居（继续 1 跳展开） |
| **点画布空白处 / `Esc`** | 退出聚焦、恢复完整视图（无聚焦时 `Esc` 依次：重置回放 → 重置视图） |
| 顶部 **「适应画布」**（`#btn-fit`） | 只调整缩放/平移，不改筛选与聚焦（原「重置视图」改名，避免混淆） |
| 顶部 **「重置」**（`#btn-reset`） | 清聚焦 / 搜索 / 类型过滤 / 路径 / 回放高亮，回到初始默认视图并 fit；`graph_fast.html` 若当前是全图模式则保持全图并 fit |

> 聚焦与搜索/路径/回放镜头互斥：开始搜索、查找路径、回放步进会自动退出聚焦，避免高亮叠加。

## 4. `graph_fast.html`（WebGL 性能版）

- **默认只渲染当前算子的 2 跳子图**（默认 gemm：60 节点 / 254 边）；顶部「算子」下拉可切换 12 个算子。
- **「加载全图」**：862 节点 / 3656 边由 WebGL 渲染，实测切换 ≈3ms、单帧 CPU 提交 ≈2.9ms；
  **「回到子图」** 恢复。子图模式默认显示标签，全图模式标签只在 hover / 选中时显示。
- 交互：类型过滤、搜索（防抖 + 高亮）、**点击节点聚焦（中心+1 跳邻居，其余 dimColor 变暗、隐藏标签；点空白/Esc/重置退出）**、点击详情（属性 + 来源行号 + 关联证据 + 关联节点分组 + 展开邻居）、
  BFS 路径（自动加载子图外路径节点 + 相机跟随）、7 步检索回放（上一步/下一步/自动播放/重置 + reason/dropped）。

## 5. 性能优化要点（详见 [PERF.md](PERF.md)）

| 问题（旧版） | 解决 |
|---|---|
| vis 物理引擎 stabilize 862 节点（首屏 ≈1.7s） | 预计算确定性布局（≈25ms）+ `physics:false` |
| 每次 hover/选中都全量更新 862 节点 + 3656 边（≈122ms） | 增量 diff：选中只更新前后 2 个节点；hover 只更新标签 |
| 862 个标签 + 3656 条曲线边每帧重绘 | 标签按需（选中/hover/zoom>1.3）；边 `smooth=false`；拖拽隐藏边/节点 |
| 默认渲染全部 862 节点 | 默认只渲染 4 类核心实体（614）+ 「展开 1 跳邻居」懒加载 + 「显示全图」开关 |
| Canvas 逐节点绘制 | `graph_fast.html` 用 Sigma.js WebGL（GPU 合成） |

## 6. 深链接（两页均支持 `selftest` / `perfprobe` / `bench`）

| 参数 | 说明 | 示例 |
|---|---|---|
| `panel` | `detail` / `path` / `chain` / `replay`（chain 仅 graph.html） | `?panel=replay` |
| `select` | 预选中节点 id | `?panel=chain&select=stg:abstract:k_unroll` |
| `trace` / `step` | 指定 trace / 回放到第 N 步（1 起） | `?panel=replay&step=4` |
| `from` / `to` | 路径起点/终点（自动 BFS） | `?panel=path&from=op:gemm&to=art:axpy:analysis` |
| `q` | 预填搜索词 | `?q=gemm` |
| `show_all` | `1` = 直接显示全图（graph.html） | `?show_all=1` |
| `full` | `1` = 直接加载全图（graph_fast.html） | `?full=1` |
| `op` | 指定算子 2 跳子图（graph_fast.html） | `?op=op:gemm` |
| `renderer` | `vis` / `svg`（graph.html 降级渲染） | `?renderer=svg` |
| `focus` | 直接聚焦到指定节点 | `?focus=op:gemm` |
| `reset` | `1` = 先按 select 聚焦再重置（截图/演示用） | `?select=op:gemm&reset=1` |
| `selftest` / `bench` / `perfprobe` | 自检 / 性能基准 / 打印性能指标 | `?selftest=1` |

## 7. 数据接入（pipeline 侧）

1. `python -m kgagent.cli export-viz` 生成 `viz/graph_data.js` 与 `viz/traces.js`：
   - `graph_data.js`：`window.KG_DATA = {nodes:[...], edges:[...], stats:{...}};`
   - `traces.js`：`window.KG_TRACES = {<trace_id>: {trace_id, query, steps[], result}};`
2. 刷新页面即用；顶部徽标会从「示例数据」变为「真实数据」，统计自动更新。

字段约定见 `../SPEC.md` §2：节点必须有 `id/type/name/properties/provenance`
（`provenance.source_file` 必填）；边必须有 `id/type/head/tail`；trace 每步必须有
`name/description/candidates/selected`，淘汰项必须有 `dropped[].reason`。
12 类实体、14 类关系均有样式映射，未知名自动落「其他」灰色，不会报错。

## 8. 验收截图（1680×1000，Chrome headless 实拍，当前 862 节点数据）

| 截图 | 内容 |
|---|---|
| `_screenshots/01_graph.png` | graph.html 性能模式（614/862 节点，详情面板 + 折叠邻居提示） |
| `_screenshots/02_replay_step4.png` | 检索回放第 4 步（候选/选中/淘汰 + dropped 原因） |
| `_screenshots/03_chain.png` | 推理链 5 卡片 |
| `_screenshots/04_path_long.png` | 路径跳转（BFS 高亮 + 关系列表） |
| `_screenshots/06_search.png` | 搜索高亮 |
| `_screenshots/fast_subgraph.png` | graph_fast.html 默认算子 2 跳子图（60 节点，WebGL） |
| `_screenshots/fast_full.png` | graph_fast.html 加载全图（862 节点 / 3656 边，WebGL） |
| `_screenshots/old_full_vis.png` | 优化前旧版（862 节点全量 vis，首屏 ≈1.7s）对比用 |
| `_screenshots/focus_graph.png` | graph.html 点击 GEMM 后聚焦（6 邻居高亮、其余变暗、详情按关系分组） |
| `_screenshots/focus_fast.png` | graph_fast.html 点击 GEMM 后聚焦（中心+邻居、其余 dimColor） |
| `_screenshots/reset_after_focus.png` | 聚焦后点「重置」→ 回到初始性能模式视图 |

## 9. 已知限制

1. headless 环境的 FPS 不可作为真实帧率（WebGL 走 SwiftShader 软件光栅化）；真实 GPU 浏览器体验需本机人工确认。
2. `graph_fast.html` 为精简 UI（详情/路径/回放，无推理链卡片；推理链请用 `graph.html`）。
3. WebGL 需浏览器硬件加速；极老机器可能初始化失败，页面会提示改用 `graph.html`。
4. 性能模式下非核心类型默认折叠，搜索/路径/回放命中时会自动展开；纯"图探索"需点「展开 1 跳邻居」。
5. 页面只读 `viz/**` 下脚本，不读取 `../data/`，也不联网。

## 10. 验收清单

- [x] `graph.html` 双击可开（file://），不联网，中文正常；`?selftest=1` **21/21**（含聚焦/退出/重置断言）
- [x] 性能模式：默认 614/862 节点、关物理、标签按需、拖拽隐藏边/节点、边直线
- [x] 懒加载：详情「展开 1 跳邻居」+ 顶部「显示全图/回到性能模式」开关
- [x] 五项交互保留：图探索 / 路径 / 推理链 / 检索回放 / 搜索+图例
- [x] 点击聚焦（节点+1 跳邻居高亮、其余变暗、非相关标签隐藏）+ 详情「关联节点」分组 + 点空白/Esc 退出 + 「重置」按钮；`#btn-fit` 改名「适应画布」
- [x] `graph_fast.html`：Sigma.js WebGL 本地加载；默认 2 跳子图；加载全图 862/3656 可交互；`?selftest=1` **15/15**（含聚焦/重置）
- [x] 首屏：graph.html ≈306ms（旧版 ≈1684ms）；graph_fast ≈544ms（<2s）
- [x] 只写 `kg_agent/viz/**`；`PERF.md` 记录前后对比、切换方式、限制
