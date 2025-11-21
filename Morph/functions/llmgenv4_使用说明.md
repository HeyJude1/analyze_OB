# llmgenv4.py 使用说明

## 概述

`llmgenv4.py` 是基于 `llmgenv3.py` 的增强版本，主要新增了从Milvus数据库查询优化策略实体详情的功能，能够生成更丰富、更准确的优化策略说明。

## 主要改进

### 1. 🔗 Milvus集成
- 自动连接Milvus数据库
- 根据策略UID查询实体详细信息
- 获取 `rationale`、`implementation`、`impact` 等关键字段
- 支持优雅降级（Milvus不可用时使用基础信息）

### 2. 📋 增强的策略信息
相比v3版本，v4版本的优化策略包含：

**基础信息（v3已有）**：
- 策略名称和UID
- 核心模式和上下文模式
- 优化得分和模式频次

**新增详细信息（v4新增）**：
- **Level**: 优化级别（algorithm/code/instruction）
- **Rationale**: 策略的理论依据和原理
- **Implementation**: 具体的实现方法和技巧
- **Impact**: 预期的性能提升效果
- **Trade-offs**: 权衡考虑和潜在问题

### 3. 🛠️ 灵活配置
- 支持自定义Milvus配置文件路径
- 支持禁用Milvus功能（`--no-milvus`）
- 兼容原有的所有参数

## 使用方法

### 基本用法

```bash
cd /home/dgc/mjs/project/analyze_OB/Morph

# 使用默认配置
python functions/llmgenv4.py

# 指定输入和输出文件
python functions/llmgenv4.py \
    --input prompts1.json \
    --output results/blas_optimized_code_v4.json
```

### 完整参数

```bash
python functions/llmgenv4.py \
    --input prompts1.json \
    --output results/blas_optimized_code_v4.json \
    --strategy_dir /home/dgc/mjs/project/analyze_OB/op_results \
    --config ../KG/kg_config.json \
    --model qwen-plus-2025-04-28 \
    --temperature 0.0 \
    --max_tokens 1024 \
    --dry  # 干运行模式，不实际生成代码
```

### 禁用Milvus功能

如果Milvus不可用或只想使用基础策略信息：

```bash
python functions/llmgenv4.py \
    --input prompts1.json \
    --output results/blas_basic_code.json \
    --no-milvus
```

## 参数说明

| 参数 | 默认值 | 说明 |
|-----|--------|------|
| `--input` | `prompts1.json` | 输入prompt文件 |
| `--output` | `results/prompts_code.json` | 输出代码文件 |
| `--strategy_dir` | `/home/dgc/mjs/project/analyze_OB/op_results` | 优化策略目录 |
| `--config` | `../../KG/kg_config.json` | Milvus配置文件 |
| `--model` | `qwen-plus-2025-04-28` | LLM模型名称 |
| `--temperature` | `0.0` | 生成温度 |
| `--top_p` | `0.9` | Top-p采样 |
| `--max_tokens` | `1024` | 最大生成token数 |
| `--dry` | `False` | 干运行模式 |
| `--overwrite` | `False` | 覆盖现有输出 |
| `--no-milvus` | `False` | 禁用Milvus功能 |

## 输出格式对比

### v3版本的策略信息
```
Recommended optimization strategies:

1. **Loop Unrolling Strategy**
   - Core patterns: loop_unroll, vectorization
   - Contextual patterns: cache_optimization
   - Optimization score: 0.85
   - Implementation: Use pragma unroll
   - Expected impact: 20% performance gain
```

### v4版本的策略信息（增强）
```
Recommended optimization strategies:

1. **Loop Unrolling Strategy**
   - Level: algorithm
   - Rationale: Loop unrolling reduces branch overhead and enables better instruction-level parallelism by executing multiple iterations in a single loop body
   - Implementation: Apply #pragma unroll with factor 4-8 for inner loops, ensure register pressure doesn't exceed limits
   - Impact: 15-25% performance improvement for compute-bound kernels with regular access patterns
   - Trade-offs: Increased code size may affect instruction cache, requires careful register management
   - Core patterns: loop_unroll, vectorization
   - Contextual patterns: cache_optimization
   - Optimization score: 0.85
```

## 工作流程

### 1. 初始化阶段
```
🔗 初始化Milvus连接...
✅ 已连接到Milvus: localhost:19530/code_op
```

### 2. 策略加载阶段
对每个算子：
1. 读取策略JSON文件（如 `01_gemm/01_gemm.json`）
2. 提取 `final_strategies` 列表
3. 对每个策略，根据UID查询Milvus获取详细信息
4. 合并JSON和Milvus信息，生成丰富的策略描述

### 3. 代码生成阶段
1. 构建包含增强策略信息的prompt
2. 调用LLM生成优化代码
3. 后处理和保存结果

### 4. 清理阶段
```
✅ Milvus连接已关闭
🎉 代码生成完成！结果保存至: results/blas_optimized_code_v4.json
```

## 错误处理

### Milvus连接失败
```
⚠️ 警告: Milvus连接失败，将使用基础策略信息: Connection refused
```
- 自动降级到基础模式
- 仍可正常生成代码，但策略信息较少

### 策略文件不存在
```
⚠️ 警告: 优化策略文件不存在: /path/to/strategy.json
```
- 使用默认策略信息
- 不影响代码生成流程

### 实体查询失败
```
⚠️ 警告: 未找到UID为 abc123 的实体
```
- 使用JSON文件中的基础信息
- 继续处理其他策略

## 测试和验证

### 运行测试脚本
```bash
cd /home/dgc/mjs/project/analyze_OB/Morph
python test_llmgenv4.py
```

测试内容：
1. **策略文件检查**: 验证JSON文件格式和内容
2. **Milvus连接**: 测试数据库连接和查询
3. **策略加载**: 对比基础版本和增强版本

### 预期输出
```
🧪 llmgenv4.py Milvus集成功能测试
============================================================
🔍 检查策略文件内容...
📊 策略文件统计:
   - 文件大小: 15234 字节
   - final_strategies 数量: 3
   - 第一个策略字段: ['uid', 'name', 'core_patterns', ...]
   - 第一个策略UID: abc123-def456-789

🔗 测试Milvus连接...
✅ 已连接到Milvus: localhost:19530/code_op
✅ Milvus连接成功

📋 测试优化策略加载...
📄 测试基础策略加载（不使用Milvus）...
✅ 基础策略加载成功，长度: 1234 字符
🔗 测试增强策略加载（使用Milvus）...
✅ 增强策略加载成功，长度: 2345 字符
📈 Milvus增强版本增加了 1111 个字符的详细信息

============================================================
🏁 测试结果总结:
   - 策略文件检查: ✅ 通过
   - Milvus连接: ✅ 通过
   - 策略加载: ✅ 通过

🎉 所有测试通过！llmgenv4.py 已准备就绪
```

## 性能优化建议

### 1. Milvus连接复用
- 在批量处理时，一次连接处理所有算子
- 避免频繁连接/断开

### 2. 缓存查询结果
- 对相同UID的实体，缓存查询结果
- 减少重复的数据库查询

### 3. 并行处理
- 可以并行查询多个实体的详情
- 但要注意Milvus连接数限制

## 故障排除

### 常见问题

1. **ImportError: No module named 'pymilvus'**
   ```bash
   pip install pymilvus
   ```

2. **连接超时**
   - 检查Milvus服务是否运行
   - 验证 `kg_config.json` 中的连接配置

3. **权限错误**
   - 确保有读取配置文件的权限
   - 检查输出目录的写入权限

4. **内存不足**
   - 减少批处理大小
   - 使用 `--dry` 模式测试

### 调试模式

启用详细日志：
```bash
export PYTHONPATH=/path/to/project
python -v functions/llmgenv4.py --dry
```

## 总结

llmgenv4.py 通过集成Milvus数据库，显著增强了优化策略的详细程度和准确性，为LLM生成更高质量的优化代码提供了强有力的支持。

**主要优势**：
- 🔍 **更丰富的策略信息**: 从Milvus获取完整的实体详情
- 🛡️ **健壮的错误处理**: 优雅降级，确保系统稳定性
- ⚙️ **灵活的配置**: 支持多种使用场景
- 🧪 **完善的测试**: 提供全面的功能验证

这为基于知识的高性能计算代码生成奠定了坚实的基础！
