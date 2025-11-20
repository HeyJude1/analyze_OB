from functions import JSON2CSV, MetricsCSV

if __name__ == "__main__":
    # CodeGen(input_path="prompts.json", output_path="results/code.json")
    # run-all生成code_run.json
    # CodeRe(input_path="results/code_run.json", output_path="results/code_review.json")
    # JSON2CSV(json_path="results/code_review_run.json", csv_path="results/code_review_run.csv")
    # MetricsCSV(input_csv="prompt_test/promptsv1_code_run.csv")
    JSON2CSV(json_path="/home/dgc/wangzicong/DeepThink/code2.json", csv_path="results/conf_code_run.csv")
    MetricsCSV(input_csv="results/conf_code_run.csv")
    