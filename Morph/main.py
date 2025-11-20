"""
用于运行函数
"""
from functions import CodeGenv1, CodeRe, CodeGenv2, CodeGenv3
# from drivers import RunCode

if __name__ == "__main__":
    # 利用大模型生成代码
    # CodeGen(prompts_path="data/prompts.json", outputs_path="results/prompts_code.json")

    # 利用驱动测试代码

    # 将无法运行的代码进行二次生成
    # CodeRe(input_path="results/code_run.json", output_path="results/code_review.json", model="qwen-plus-2025-04-28")

    # 再次运行二次生成的代码(这里可以加上循环，进行多次生成，直到运行成功为止)
    CodeGenv1(input_path="prompts.json", output_path="prompt_test/promptsv1_code.json", temperature=0.2)
    CodeGenv1(input_path="prompts.json", output_path="prompt_test/promptsv1_code_t0.json", temperature=0)
    # CodeGenv2(input_path="prompts.json", output_path="prompt_test/promptsv2_code_t0.json")
    # CodeGenv3(input_path="prompts.json", output_path="prompt_test/promptsv3_code_t0.json")


    




        




    


    



