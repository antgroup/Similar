import os
import torch
import numpy as np
from safetensors.torch import load_file
from argparse import ArgumentParser
from tqdm.auto import tqdm
from scipy.stats import spearmanr
import pandas as pd
from glob import glob
from torch import nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import datasets

# Enable TF32 for improved performance on Ampere GPUs
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Define the attributes for multi-objective reward modeling  定义多目标奖励建模的属性
attributes = [
    "IP",
    "E",
    "TC",
    "TR",
    "C",
]


class GatingNetwork(nn.Module):
    """
    Gating Network: A simple MLP with softmax output and temperature scaling
    This network learns to combine multiple reward objectives based on the input context
    门控网络：一种具有softmax输出和温度缩放的简单MLP
    该网络学习根据输入上下文组合多个奖励目标
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        temperature: float = 10,
        logit_scale: float = 1.0,
        hidden_dim: int = 1024,
        n_hidden: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.temperature = temperature
        self.logit_scale = nn.Parameter(torch.ones(1) * logit_scale)
        self.dropout_prob = dropout
        layers = []
        for _ in range(n_hidden):
            layers.append(nn.Linear(in_features, hidden_dim))
            in_features = hidden_dim
        layers.append(nn.Linear(in_features, out_features, bias=bias))
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        # Apply the linear layers with ReLU and dropout  使用ReLU和dropout应用线性层
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = F.relu(x)
                if self.dropout_prob > 0:
                    x = F.dropout(x, p=self.dropout_prob, training=self.training)
        # Apply softmax with temperature scaling  应用具有温度缩放功能的softmax
        x = F.softmax(x / self.temperature, dim=1)
        return x * self.logit_scale[0]


def find_proper_verbosity_penalties(cluster_V, verbosity_dim=4, corr_threshold=0.028):
    """
    Find appropriate penalties for verbosity to reduce its correlation with other dimensions
    找到对冗长的适当惩罚，以减少其与其他维度的相关性
    """
    verbosity_penalties = [
        0,
        0.01,
        0.025,
        0.05,
        0.075,
        0.1,
        0.125,
        0.15,
        0.2,
        0.25,
        0.3,
        0.35,
        0.4,
        0.45,
        0.5,
        0.55,
        0.6,
        0.65,
        0.7,
        0.75,
        0.8,
        0.85,
        0.9,
        0.95,
    ]
    verbosity_penalties = sorted(verbosity_penalties)
    K = cluster_V.shape[1]
    candidate_dims = set(range(K))
    candidate_dims.remove(verbosity_dim)
    dimwise_verbosity_penalties = np.ones(K)
    dimwise_corr = np.ones(K)
    for verbosity_penalty in verbosity_penalties:
        if len(candidate_dims) == 0:
            break
        V_adjusted = cluster_V - verbosity_penalty * cluster_V[:, [verbosity_dim]]
        corrs = {
            i: spearmanr(V_adjusted[:, i], cluster_V[:, verbosity_dim])[0]
            for i in candidate_dims
        }
        for dim, corr in corrs.items():
            if corr <= corr_threshold:
                candidate_dims.remove(dim)
                dimwise_verbosity_penalties[dim] = verbosity_penalty
                dimwise_corr[dim] = corr
            else:
                dimwise_corr[dim] = np.min([dimwise_corr[dim], corr])
        if len(candidate_dims) == 0:
            break
    return {"penalty": dimwise_verbosity_penalties, "corr": dimwise_corr}


def load_embeddings(embedding_path_pattern, device):
    """
    Load embeddings from safetensors files  从safetensor文件加载嵌入
    """
    # Examine if the embedding path pattern is correct
    file_paths = glob(embedding_path_pattern)
    if len(file_paths) == 0:
        raise ValueError(f"Embeddings not found at {embedding_path_pattern}")
    embeddings, prompt_embeddings = [], []
    for embedding_path in file_paths:
        embeddings_data = load_file(embedding_path)
        embeddings.append(embeddings_data["embeddings"].to(device))
        prompt_embeddings.append(embeddings_data["prompt_embeddings"].to(device))
    embeddings = torch.cat(embeddings, dim=0).float()
    prompt_embeddings = torch.cat(prompt_embeddings, dim=0).float()
    return embeddings, prompt_embeddings


# Set up argument parser
parser = ArgumentParser()
parser.add_argument("--model_path", type=str, default="/mnt/prev_nas/virtual_agent/models/Qwen/Qwen2-VL-7B-Instruct")
parser.add_argument(
    "--multi_objective_dataset",
    type=str,
    default="/mnt/prev_nas/virtual_agent/MBC/Reward_Model_csv_zeta/Reward_Model_multi-objective_dataset.csv",
)
parser.add_argument(
    "--preference_dataset", type=str, default="/mnt/prev_nas/virtual_agent/MBC/Reward_Model_csv_zeta/Reward_Model_preference_dataset.csv"
)
parser.add_argument(
    "--reference_dataset",
    type=str,
    default=None,
)
parser.add_argument("--device", type=int, default=1)
parser.add_argument("--learning_rate", type=float, default=1e-3)
parser.add_argument("--weight_decay", type=float, default=0)
parser.add_argument("--n_steps", type=int, default=10000)
parser.add_argument("--batch_size", type=int, default=1024)
parser.add_argument(
    "--verbosity_dim", type=int, default=4, help="Dimension of the verbosity attribute"
)
parser.add_argument(
    "--corr_threshold",
    type=float,
    default=0.03,
    help="Correlation threshold for verbosity penalty",
)
parser.add_argument("--model_family", type=str, default="qwen2", help="Model family")
parser.add_argument("--logit_scale", type=float, default=1)
parser.add_argument("--temperature", type=float, default=10)
parser.add_argument("--n_hidden", type=int, default=3)
parser.add_argument("--hidden_size", type=int, default=1024)
parser.add_argument("--dropout", type=float, default=0.2)
args = parser.parse_args()

# Define default paths
HOME = os.path.expanduser("~")

if args.reference_dataset is None:
    args.reference_dataset = args.preference_dataset
    print(
        f"Using {args.preference_dataset} as the reference dataset for verbosity debiasing."
    )

args.model_name = args.model_path.split("/")[-1]
args.multi_objective_dataset_name = args.multi_objective_dataset.split("/")[-1]
args.preference_dataset_name = args.preference_dataset.split("/")[-1]
args.reference_dataset_name = args.reference_dataset.split("/")[-1]

args.embedding_path = "/mnt/prev_nas/virtual_agent/MBC/RLHF-Reward-Modeling/checkpoints/gating_network_prepare_3-train.safetensors"
args.regression_layer_path = "/mnt/prev_nas/virtual_agent/MBC/RLHF-Reward-Modeling/checkpoints/Qwen2-VL-7B-Instruct_Reward_Model.pt"

device = f"cuda:{args.device}" if args.device >= 0 else "cpu"

# Print the paths for verification
print(f"Embedding path: {args.embedding_path}")
print(f"Regression layer path: {args.regression_layer_path}")

# Load embeddings
print("Loading embeddings...")
embeddings, prompt_embeddings = load_embeddings(args.embedding_path, device=device)

# Load regression layer  加载回归层
print("Loading regression layer...")
regression_layer = torch.load(args.regression_layer_path, map_location=device)["weight"]

n_attributes, hidden_size = regression_layer.shape

# Load reference dataset embeddings  加载参考数据集嵌入
embedding_path = "/mnt/prev_nas/virtual_agent/MBC/RLHF-Reward-Modeling/checkpoints/gating_network_prepare_3-train.safetensors"
ref_embeddings, ref_prompt_embeddings = load_embeddings(embedding_path, device=device)

# Calculate pairwise rewards and rewards difference  计算成对奖励和奖励差异
pairwise_rewards = ref_embeddings @ regression_layer.T
rewards = pairwise_rewards.reshape(-1, pairwise_rewards.shape[-1])
rewards_diff = pairwise_rewards[:, 0] - pairwise_rewards[:, 1]

# Find proper verbosity penalties  找到适当的冗长处罚
penalties = find_proper_verbosity_penalties(
    rewards.cpu().numpy().reshape(-1, n_attributes),
    verbosity_dim=args.verbosity_dim,
    corr_threshold=args.corr_threshold,
)
print("Penalties:", penalties)

# Create reward transform matrix  创建奖励转换矩阵
reward_transform_matrix = torch.eye(n_attributes)
reward_transform_matrix[args.verbosity_dim, :] -= torch.from_numpy(penalties["penalty"])
reward_transform_matrix = reward_transform_matrix.to(device)

# Prepare data for training  准备训练数据
X = prompt_embeddings  # condition for gating network
Z = embeddings  # multi-objective rewards
R = embeddings @ regression_layer.T @ reward_transform_matrix  # multi-objective rewards
# Split train/val
X_train, X_val, Z_train, Z_val, R_train, R_val = train_test_split(
    X, Z, R, test_size=0.2, random_state=0
)

# Initialize gating network  初始化门控网络
print("Initializing gating network...")
gating_network = GatingNetwork(
    X_train.shape[-1],
    regression_layer.shape[0],
    n_hidden=args.n_hidden,
    hidden_dim=args.hidden_size,
    logit_scale=args.logit_scale,
    temperature=args.temperature,
    dropout=args.dropout,
).to(device)

# Define loss function and optimizer  定义损失函数和优化器
loss_fn = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(
    gating_network.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.n_steps)

# Training loop
print("Starting training...")
for step in tqdm(range(args.n_steps)):
    optimizer.zero_grad()

    # Sample batch
    idx = torch.randint(0, X_train.shape[0], (args.batch_size,))
    X_batch = X_train[idx]
    Z_batch = Z_train[idx]

    # Forward pass
    gating_weights = gating_network(X_batch)
    pred = torch.sum(Z_batch @ regression_layer.T * gating_weights, dim=-1)

    # Compute loss
    loss = loss_fn(pred[:, 0] - pred[:, 1], torch.ones_like(pred[:, 0]))

    # Backward pass and optimization
    loss.backward()
    optimizer.step()
    scheduler.step()

    if step % 100 == 0:
        print(f"Step {step}, Loss: {loss.item():.4f}")

# Evaluation
print("Evaluating model...")
gating_network.eval()
with torch.no_grad():
    gating_weights_val = gating_network(X_val)
    pred_val = torch.sum(Z_val @ regression_layer.T * gating_weights_val, dim=-1)
    acc_val = ((pred_val[:, 0] - pred_val[:, 1]) > 0).float().mean()
    print(f"Validation accuracy: {acc_val.item():.4f}")

# Save the trained gating network  存储训练好的门控网络
save_path = "/mnt/prev_nas/virtual_agent/MBC/RLHF-Reward-Modeling/checkpoints/gating_network_train.pt"
torch.save(gating_network.state_dict(), save_path)
print(f"Saved gating network to {save_path}")
