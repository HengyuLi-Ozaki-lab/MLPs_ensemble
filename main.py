import pandas as pd
from utils.input_parser import load_and_split_traj
from model.ensembled_output import ModelManager

# Step 1: 加载数据集并划分训练集和测试集
traj_file = "/home/lee/mlps/internship/Dataset-selected/dataset-6k-selected_20241209.traj"  # 替换为实际路径
output_dir = "/home/lee/mlps/internship/Dataset-selected"
dataset = load_and_split_traj(traj_file=traj_file, output_dir=output_dir, test_size=1)
test_data = dataset["test"]


# Model configuration
model_configs = [
    {"name": "MACE", "params": {"model": "large", "device": "cpu"}},
    {"name": "EqV2", "params": {"pretrained_path": "pretrained_models"}},
    {"name": "CHGNET", "params": {}},
    {"name": "MatterSim", "params": {"load_path": "MatterSim-v1.0.0-5M.pth"}},
]

manager = ModelManager(model_configs)

# Step 4: 执行批量预测
print("Starting batch prediction on test set...")
results = manager.predict_batch(test_data,prediction_type='energy')

# Step 5: 记录结果
results_df = pd.DataFrame(results)

# 添加真实能量列
true_energies = [data["true_energy"] for data in test_data]
results_df["True Energy"] = true_energies

# 保存结果
output_file = "/app/6k_predictions.csv"
results_df.to_csv(output_file, index=False)
print(f"Test set predictions saved to {output_file}.")

# 打印部分结果
print(results_df.head())

'''
from utils.input_parser import InputParser
from utils.ensembled_output import ModelManager

def main():
    input_dir = "/app/MLP_ensemble_source/test/"
    output_file = "/app/MLP_ensemble_source/test/unified_results.json"

    # Model configuration
    model_configs = [
        {"name": "MACE", "params": {"model": "large", "device": "cpu"}},
        {"name": "EqV2", "params": {"pretrained_path": "pretrained_models"}},
        {"name": "CHGNET", "params": {}},
        {"name": "MatterSim", "params": {"load_path": "MatterSim-v1.0.0-5M.pth"}},
    ]

    # Initial
    parser = InputParser()
    manager = ModelManager(model_configs)

    try:
        # Step 1: Batch load structure
        print("Loading structures...")
        parsed_data = parser.batch_parse(input_dir, file_extensions=["cif", "xyz", "poscar", "vasp"])

        # Step 2: Batch prediction
        print("Starting unified batch prediction...")
        results = manager.predict_batch(parsed_data,prediction_type='energy')

        print(results)

        # Step 3: Save results
        #print(f"Saving results to {output_file}...")
        #manager.save_results(results, output_file)

        #print("Unified batch prediction completed successfully.")

    except Exception as e:
        print(f"Unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
'''