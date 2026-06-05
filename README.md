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

```
openblas算子源码  →  LLM分析(算法/代码/指令三层)  →  实体抽取  →  消歧聚类  →  知识图谱(Milvus/Neo4j)
                                                                              ↓
新代码  →  特征提取  →  相似度检索  →  优化策略推荐
```

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
