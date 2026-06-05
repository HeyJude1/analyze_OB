# OpenBLAS 优化策略知识图谱

从 OpenBLAS 编译产出的算子代码中提取优化策略，通过大模型进行实体抽取和消歧，构建知识图谱，支持对新代码的优化策略推荐。

## 项目结构

```
analyze_OB_v1/
├── src/                          # 核心源码
│   ├── analysis/                 # 优化策略分析模块
│   │   ├── optimizers.py         # 三层优化器 (算法/代码/指令)
│   │   └── workflow.py           # LangGraph 工作流编排
│   ├── kg/                       # 知识图谱模块
│   │   ├── extractor.py          # 实体抽取器
│   │   ├── cluster.py            # 实体聚类检索
│   │   ├── refine.py             # 聚类精炼
│   │   ├── merger.py             # 关系合并
│   │   ├── alignment.py          # 实体对齐调度
│   │   ├── retrieval.py          # 优化策略检索与评分
│   │   ├── export.py             # Milvus → Neo4j 导出
│   │   └── backup.py             # 数据库备份
│   └── utils/                    # 共享工具
├── config/                       # 配置文件
│   ├── config.json               # 主配置
│   ├── kg_config.json            # 知识图谱配置 (Milvus/Neo4j)
│   └── .env                      # API 密钥
├── data/                         # 输入数据
│   └── openblas-output/          # OpenBLAS 各架构算子源码
├── prompts/                      # Prompt 模板
├── output/                       # 生成物
│   ├── results/                  # 分析结果
│   ├── op_results/               # 按算子组织的结果
│   └── strategy_reports/         # 策略报告
├── morph/                        # 代码生成实验项目
├── scripts/                      # 入口脚本
├── docs/                         # 文档
│   ├── design/                   # 设计方案
│   ├── instrument/               # OpenBLAS 分析说明
│   └── papers/                   # 参考论文
├── tests/                        # 测试
└── README.md
```

## 流程

### 1. 知识图谱构建

```
OpenBLAS算子源码  →  LLM分析(算法/代码/指令三层)  →  实体抽取  →  消歧聚类  →  知识图谱(Milvus/Neo4j)
```

**Step 1: 优化策略分析**
```bash
python -m src.analysis.workflow
# 输入: data/openblas-output/ 下的算子源码
# 输出: output/results/ 下的分析结果 JSON
```

**Step 2: 实体抽取**
```bash
python src/kg/extractor.py --config config/kg_config.json --data_dir output/results/<timestamp>
# 从分析结果中提取实体和关系，写入 Milvus
```

**Step 3: 实体对齐(消歧)**
```bash
python src/kg/alignment.py --rounds 3 --config config/kg_config.json
# 聚类 → 精炼 → 合并，3轮迭代消除重复实体
```

**Step 4: 导出 Neo4j**
```bash
python src/kg/export.py --config config/kg_config.json
# 将 Milvus 中的图谱数据导出到 Neo4j 图数据库
```

### 2. 优化策略推荐

对一段新的算子代码，从知识图谱中检索匹配的优化策略：

```bash
python src/kg/retrieval.py --source your_code.c --config config/kg_config.json
```

推荐流程：
```
新代码  →  四阶段计算流程识别  →  Milvus向量相似度检索  →  关联策略查找  →  评分排序  →  输出推荐策略
```

四阶段识别器（`retrieval.py` 内置）自动分析输入代码的：
1. **计算准备** — 参数校验、索引初始化、循环不变量计算
2. **数据转换** — 打包/解包、转置
3. **核心计算** — 向量归约、矩阵乘法、微内核、分块循环、三角求解
4. **后处理** — 结果缩放与写回

每个识别到的计算流程会从 Milvus 中检索语义最相似的已知优化策略，并通过 LLM 评估适用性后按评分排序输出。

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 配置 API 密钥
# 编辑 config/.env: DASHSCOPE_API_KEY=your_key

# 运行完整分析
bash scripts/run_full_analysis.sh

# 单独运行分析工作流
python -m src.analysis.workflow

# 运行知识图谱提取
python src/kg/extractor.py --config config/kg_config.json

# 运行实体对齐
python src/kg/alignment.py --rounds 3 --config config/kg_config.json

# 导出到 Neo4j
python src/kg/export.py --config config/kg_config.json
```

## 配置

- `config/config.json` — 模型参数、数据路径
- `config/kg_config.json` — Milvus/Neo4j 连接、聚类阈值、实体对齐参数
- `config/.env` — `DASHSCOPE_API_KEY`
