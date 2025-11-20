"""
统计数据的函数
"""
# std imports
import argparse
import json

# third party imports
import pandas as pd

def has_outputs(prompt: dict) -> bool:
    """ 检查一个 prompt 是否包含有效的 outputs 字段 """
    if "outputs" not in prompt:
        return False
    if not isinstance(prompt["outputs"], list) or len(prompt["outputs"]) == 0:
        return False
    if all(isinstance(o, str) for o in prompt["outputs"]):
        return False
    return all(isinstance(o, dict) for o in prompt["outputs"])

def check(df: pd.DataFrame):
    """ 检查 (name, parallelism_model) 对是否有 0 个成功编译的情况。这有助于发现代码生成失败的问题。"""
    agg = df.groupby(["name", "parallelism_model"]).agg({"did_build": "sum"})
    agg = agg[agg["did_build"] == 0]
    if len(agg) > 0:
        print("The following (name, parallelism_model) pairs have zero successful builds:")
        print(agg)

# 将json文件转为csv文件
def JSON2CSV(json_path, csv_path) -> None:
    # 加载json文件
    with open(json_path, "r") as f:
        input_json = json.load(f)
    
    # 过滤没有output的json文件
    input_json = list(filter(lambda x: has_outputs(x), input_json))

    rows = [] # 用于保存扁平化后的每一行数据

    # 遍历每一个prompt，每个 prompt 可能有多个 outputs
    for prompt in input_json:
        for output_idx, output in enumerate(prompt["outputs"]):
            if output["runs"] is None:
                row = {
                    "prompt": prompt["prompt"],
                    "name": prompt["name"],
                    "problem_type": prompt["problem_type"],
                    "language": prompt["language"],
                    "parallelism_model": prompt["parallelism_model"],
                    "temperature": prompt["temperature"],
                    "top_p": prompt["top_p"],
                    "do_sample": prompt["do_sample"],
                    "max_new_tokens": prompt["max_new_tokens"],
                    "prompted": prompt.get("prompted", False),
                    "generated_output": output["generated_output"],
                    "did_build": output["did_build"],
                    "build_stderr": output["build_stderr"], # 加上了错误信息
                    "is_source_valid": output["is_source_valid"],
                    "best_sequential_runtime": output["best_sequential_runtime"],
                    "output_idx": output_idx 
                }
                rows.append(row)
                continue
            
            # 有运行记录，将每次运行也展开成单独行
            for run_idx, run in enumerate(output["runs"]):
                row = {
                    "prompt": prompt["prompt"],
                    "name": prompt["name"],
                    "problem_type": prompt["problem_type"],
                    "language": prompt["language"],
                    "parallelism_model": prompt["parallelism_model"],
                    "temperature": prompt["temperature"],
                    "top_p": prompt["top_p"],
                    "do_sample": prompt["do_sample"],
                    "max_new_tokens": prompt["max_new_tokens"],
                    "prompted": prompt.get("prompted", False),
                    "generated_output": output["generated_output"],
                    "did_build": output["did_build"],
                    "build_stderr": output["build_stderr"],
                    "is_source_valid": output["is_source_valid"],
                    "best_sequential_runtime": output["best_sequential_runtime"],
                    "output_idx": output_idx,
                    "run_idx": run_idx,
                    **run # 展开 run 中的所有字段，如 runtime、output 等
                }
                rows.append(row)

    # 创建 DataFrame 表格
    df = pd.DataFrame(rows)

    # 检查是否存在构建失败的组合
    check(df)

    # 将换行符替换为 "\\n"，防止破坏 CSV 文件格式
    df.prompt = df.prompt.apply(lambda x: x.replace("\n", "\\n"))
    df.generated_output = df.generated_output.apply(lambda x: x.replace("\n", "\\n"))

    # 写入到 CSV 文件
    df.to_csv(csv_path, index=False)

# 统计CSV表中的数据，计算构建成功率和运行成功率
def MetricsCSV(input_csv: str) -> None:
    try:
        # 读取文件
        df = pd.read_csv(input_csv)
        
        # 检查是否包含Did_build列
        # print(df.columns)
        if "did_build" not in df.columns:
            raise ValueError("CSV file does not contain 'did_build' column.")
        # 检查是否包含Did_run列
        if "did_run" not in df.columns:
            raise ValueError("CSV file does not contain 'Did_run' column.")

        # 去除空值
        did_build_series = df["did_build"].dropna()
        did_run_series = df["did_run"].dropna()
        
        # 统计
        total_count = len(df)
        total_build_count = len(did_build_series)
        total_run_count = len(did_run_series)
        true_count_build = (did_build_series == True).sum()
        true_count_run = (did_run_series == True).sum()

        # 输出结果
        print(f"文件 {input_csv} 共有 {total_count} 行数据")
        print(f"构建总数：{total_build_count}")
        print(f"运行总数：{total_run_count}")
        print(f"构建成功数量: {true_count_build}, 成功率: {true_count_build / total_build_count * 100:.2f}%")
        print(f"运行成功数量: {true_count_run}, 成功率: {true_count_run / total_run_count * 100:.2f}%")
    
    except Exception as e:
        print(f"An error occurred: {e}")


    
    
