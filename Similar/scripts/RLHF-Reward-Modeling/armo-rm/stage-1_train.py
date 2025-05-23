import os
import torch
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from safetensors.torch import load_file
from argparse import ArgumentParser


"""
Perform multi-objective linear regression on precomputed embeddings.
This script loads embeddings and labels, splits the data into training and validation sets,
trains Ridge regression models for each attribute across a range of regularization strengths (alphas),
selects the best alpha based on validation loss, and saves the resulting regression weights.
对预先计算的嵌入进行多目标线性回归。
该脚本加载嵌入和标签，将数据拆分为训练集和验证集，
在一系列正则化强度（α）范围内为每个属性训练Ridge回归模型，
根据验证损失选择最佳alpha，并保存得到的回归权重。
"""

# ---------------------------
# Argument Parsing  获取参数
# ---------------------------
parser = ArgumentParser(description="Linear Probing on Precomputed Embeddings")
parser.add_argument(
    "--model_path",
    type=str,
    default="sfairXC/FsfairX-LLaMA3-RM-v0.1",
    help="Path to the pre-trained model (HuggingFace path or local folder)",
)
parser.add_argument(
    "--dataset_path",
    type=str,
    default="RLHFlow/ArmoRM-Multi-Objective-Data-v0.1",
    help="Path to the dataset containing multi-objective labels (HuggingFace path or local folder)",
)
parser.add_argument(
    "--embeddings_dir",
    type=str,
    default=None,
    help="Path to the directory containing embedding files. If not provided, defaults to HOME/data/ArmoRM/embeddings/<model_name>/<dataset_name>",
)
parser.add_argument(
    "--output_dir",
    type=str,
    default=None,
    help="Path to save the regression weights. If not provided, defaults to HOME/data/ArmoRM/regression_weights/",
)
args = parser.parse_args()

# Extract names from paths
args.model_name = args.model_path.split("/")[-1]
args.dataset_name = args.dataset_path.split("/")[-1]

# ---------------------------
# Configuration and Setup  配置及初始化
# ---------------------------
# Define the reward attributes as per the method overview in README.md
attributes = [
    "helpsteer-helpfulness",
    "helpsteer-correctness",
    "helpsteer-coherence",
    "helpsteer-complexity",
    "helpsteer-verbosity",
    "ultrafeedback-overall_score",
    "ultrafeedback-instruction_following",
    "ultrafeedback-truthfulness",
    "ultrafeedback-honesty",
    "ultrafeedback-helpfulness",
    "beavertails-is_safe",
    "prometheus-score",
    "argilla-overall_quality",
    "argilla-judge_lm",
    "code-complexity",
    "code-style",
    "code-explanation",
    "code-instruction-following",
    "code-readability",
]

# Set the home directory
HOME = os.path.expanduser("~")

# Define the path to the embeddings based on user input or default location  根据用户输入或默认位置定义嵌入的路径
if args.embeddings_dir:
    embeddings_path = args.embeddings_dir
else:
    embeddings_path = os.path.join(
        HOME, "data", "ArmoRM", "embeddings", args.model_name, args.dataset_name
    )

# Collect all embedding files matching the pattern embeddings_path-*.safetensors  收集与模式embeddings_path-*.safetensors匹配的所有嵌入文件
embedding_files = sorted(glob(f"{embeddings_path}-*.safetensors"))

# ---------------------------
# Loading Embeddings and Labels  载入embeddings和labels
# ---------------------------
embeddings = []
labels = []
print("Loading embeddings and labels from Safetensors files...")
for file in tqdm(embedding_files, desc="Loading embeddings"):
    # Load the safetensors file
    data = load_file(file)
    embeddings.append(data["embeddings"])  # Append embeddings tensor
    labels.append(data["labels"])  # Append labels tensor

# Concatenate all embeddings and labels into single tensors  将所有嵌入和标签连接到单个张量中
embeddings = torch.cat(embeddings, dim=0).float().numpy()
labels = torch.cat(labels, dim=0).float().numpy()

print(f"Total embeddings loaded: {embeddings.shape[0]}")
print(f"Total labels loaded: {labels.shape[0]}")

# ---------------------------
# Splitting Data into Train and Validation Sets  将数据氛围训练集和验证集
# ---------------------------
print("Splitting data into training and validation sets...")
X_train, X_val, Y_train, Y_val = train_test_split(
    embeddings, labels, test_size=0.2, random_state=42
)
print(f"Training set size: {X_train.shape[0]}")
print(f"Validation set size: {X_val.shape[0]}")

# ---------------------------
# Defining Regularization Strengths (Alphas)  定义正则化优势（Alphas）
# ---------------------------
# Define a range of alpha values for Ridge regression
alphas = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]
print(f"Using alphas: {alphas}")

# Initialize a DataFrame to store the results of each Ridge regression  初始化DataFrame以存储每个Ridge回归的结果
df = pd.DataFrame(columns=["attribute", "alpha", "loss"])

# ---------------------------
# Ridge Regression Training  Ridge回归训练
# ---------------------------
print("Training Ridge regression models for each attribute and alpha...")
for attr_idx in tqdm(range(Y_train.shape[1]), desc="Attributes"):
    y_train = Y_train[:, attr_idx]
    # Create a mask to filter out NaN values in training labels  创建掩码以过滤掉训练标签中的NaN值
    valid_mask_train = ~np.isnan(y_train)
    y_train_filtered = y_train[valid_mask_train]
    X_train_filtered = X_train[valid_mask_train]

    y_val = Y_val[:, attr_idx]
    # Create a mask to filter out NaN values in validation labels  创建掩码以过滤掉验证标签中的NaN值
    valid_mask_val = ~np.isnan(y_val)
    y_val_filtered = y_val[valid_mask_val]
    X_val_filtered = X_val[valid_mask_val]

    # Iterate over each alpha to train Ridge models  迭代每个alpha以训练Ridge模型
    for alpha in tqdm(alphas, desc=f"Alpha for attribute {attr_idx}", leave=False):
        clf = Ridge(alpha=alpha, fit_intercept=False)
        clf.fit(X_train_filtered, y_train_filtered)  # Train the model
        pred = clf.predict(X_val_filtered)  # Predict on validation set
        loss = mean_squared_error(y_val_filtered, pred)  # Calculate MSE loss
        # Append the results to the DataFrame
        df = df._append(
            {"attribute": attr_idx, "alpha": alpha, "loss": loss}, ignore_index=True
        )

# ---------------------------
# Selecting Best Alphas Based on Validation Loss  基于验证损失选择最佳字母
# ---------------------------
print("Selecting the best alpha for each attribute based on validation loss...")
best_alphas = df.loc[df.groupby("attribute")["loss"].idxmin()]
print("Best alphas selected for each attribute:")
print(best_alphas)

# ---------------------------
# Fitting Final Models with Best Alphas and Extracting Weights  用最佳alpha拟合最终模型并提取权重
# ---------------------------
print(
    "Fitting final Ridge regression models with the best alphas and extracting weights..."
)
weights = []
for index, row in tqdm(
    best_alphas.iterrows(), total=best_alphas.shape[0], desc="Final Models"
):
    attr_idx = int(row["attribute"])
    best_alpha = row["alpha"]

    # Initialize Ridge model with the best alpha  使用最佳alpha初始化Ridge模型
    clf = Ridge(alpha=best_alpha, fit_intercept=False)

    # Prepare training data  准备训练数据
    y_train = Y_train[:, attr_idx]
    valid_mask_train = ~np.isnan(y_train)
    X_train_filtered = X_train[valid_mask_train]
    y_train_filtered = y_train[valid_mask_train]

    # Train the model  训练模型
    clf.fit(X_train_filtered, y_train_filtered)

    # Append the coefficient (weight) for the current attribute  为当前属性添加系数（权重）
    weights.append(clf.coef_)

    # Calculate loss on validation set for reporting  计算用于报告的验证集的损失
    y_val = Y_val[:, attr_idx]
    valid_mask_val = ~np.isnan(y_val)
    X_val_filtered = X_val[valid_mask_val]
    y_val_filtered = y_val[valid_mask_val]
    pred = clf.predict(X_val_filtered)
    loss = mean_squared_error(y_val_filtered, pred)

    print(
        f"Attribute {attr_idx} ({attributes[attr_idx]}): Best alpha = {best_alpha}, Validation Loss = {loss:.4f}"
    )

# Stack all weights into a single NumPy array  将所有权重堆叠到一个NumPy数组中
weights = np.stack(weights)
print(f"All regression weights shape: {weights.shape}")

# ---------------------------
# Saving the Regression Weights  保存回归模型的权重
# ---------------------------
# Define the output directory
if args.output_dir:
    save_dir = args.output_dir
else:
    save_dir = os.path.join(HOME, "data", "ArmoRM", "regression_weights")

os.makedirs(save_dir, exist_ok=True)  # Create the directory if it doesn't exist

# Define the path to save the weights
save_path = os.path.join(save_dir, f"{args.model_name}_{args.dataset_name}.pt")

# Save the weights as a PyTorch tensor
torch.save({"weight": torch.from_numpy(weights)}, save_path)
print(f"Saved regression weights to {save_path}")
