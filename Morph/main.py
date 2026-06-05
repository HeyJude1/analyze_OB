"""
Morph 代码生成入口
"""
from src import CodeGenv4

if __name__ == "__main__":
    # 使用 v4 生成器（支持 Milvus 优化策略集成）
    CodeGenv4(
        input_path="prompts1.json",
        output_path="results/blas_code.json",
        temperature=0.2,
    )
