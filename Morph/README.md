实验数据统计：

# 实验1（初始实验包括重生成实验）结果和中间文件在./results中：
## 1.数据集：prompts.json：omp、serial、cuda
## 2.model:全使用qwen-plus-2025-04-28
## 3.流程：
### （1）.利用llmgen中的CodeGen(prompts_path="data/prompts.json", outputs_path="results/prompts_code.json")得到prompts_code.json
### （2）.利用drivers/run-all.py测试代码得到code_run.json
### （3）.利用analysis.py中的JSON2CSV(json_path="results/code_run.json", csv_path="results/code_run.csv")得到code_run.csv
### （4）.利用analysis.py中的MetricsCSV(input_csv="results/code_run.csv")得到确定的数据
## 4.指标：
(温度0.2)
文件 prompt_test/promptsv1_code_run.csv 共有 282 行数据
构建总数：282
运行总数：216
构建成功数量: 216, 成功率: 76.60%
运行成功数量: 210, 成功率: 97.22%
# 重生成
文件 results/code_review_run.csv 共有 294 行数据
构建总数：294
运行总数：237
构建成功数量: 237, 成功率: 80.61%
运行成功数量: 231, 成功率: 97.47%
（温度0）
文件 prompt_test/promptsv1_code_t0_run.csv 共有 282 行数据
构建总数：282
运行总数：211
构建成功数量: 211, 成功率: 74.82%
运行成功数量: 196, 成功率: 92.89%

# 实验2（单纯给prompt中添加基本硬件信息）结果在./prompt_test中：
## 1.数据集：prompts.json：omp、serial、cuda
## 2.model:全使用qwen-plus-2025-04-28
## 3.流程：
### （1）.CodeGenv2(input_path="prompts.json", output_path="prompt_test/prompts_code.json")
### （2）.利用drivers/run-all.py测试代码得到prompts_code_run.json
### （3）.利用analysis.py中的JSON2CSV(json_path="prompt_test/prompts_code_run.json", csv_path="prompt_test/prompts_code_run.csv")
### （4）.利用analysis.py中的MetricsCSV(input_csv="prompt_test/prompts_code_run.csv") 
## 4.指标：
（温度0.2）
文件 prompt_test/prompts_code_run.csv 共有 246 行数据
构建总数：246
运行总数：140
构建成功数量: 140, 成功率: 56.91%
运行成功数量: 140, 成功率: 100.00%
（温度0）
文件 prompt_test/promptsv2_code_t0_run.csv 共有 255 行数据
构建总数：255
运行总数：154
构建成功数量: 154, 成功率: 60.39%
运行成功数量: 148, 成功率: 96.10%

# 实验3（给prompt中添加硬件信息得到的优化策略）结果在./prompt_test中：
## 1.数据集：prompts.json：omp、serial、cuda
## 2.model:全使用qwen-plus-2025-04-28
## 3.流程：
### （1）.CodeGenv3(input_path="prompts.json", output_path="prompt_test/promptsv3_code.json")
### （2）.利用drivers/run-all.py测试代码得到promptsv3_code_run.json
### （3）.利用analysis.py中的JSON2CSV(json_path="prompt_test/promptsv3_code_run.json", csv_path="prompt_test/promptsv3_code_run.csv")
### （4）.利用analysis.py中的MetricsCSV(input_csv="prompt_test/promptsv3_code_run.csv") 
## 4.指标：
(温度0.2)
文件 prompt_test/promptsv3_code_run.csv 共有 267 行数据
构建总数：267
运行总数：182
构建成功数量: 182, 成功率: 68.16%
运行成功数量: 176, 成功率: 96.70%
（温度0）
文件 prompt_test/promptsv3_code_t0_run.csv 共有 267 行数据
构建总数：267
运行总数：180
构建成功数量: 180, 成功率: 67.42%
运行成功数量: 179, 成功率: 99.44%

（1）在主/home/dgc/wangzicong/Morph下运行test.py：CodeGen(input_path="prompts.json",output_path="填上输出文件名") # 只保留CodeGen
（2）运行：在/home/dgc/wangzicong/Morph/drivers，运行run-all.pyn run-all.py --input_path=填上输入文件名 --output_path=填上输出文件名
（3）json转为CSV:JSON2CSV(json_path="填上输入文件名", csv_path="填上输出文件名")


