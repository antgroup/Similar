import os
import torch
import numpy as np
from safetensors.torch import load_file, save_file
from argparse import ArgumentParser
from tqdm.auto import tqdm
from scipy.stats import spearmanr
import pandas as pd
from glob import glob
from torch import nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

import datasets
from transformers import AutoTokenizer, AutoModel, AutoConfig

# Enable TF32 for improved performance on Ampere GPUs
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Define token patterns for gating different model families  定义token模式，用于gating不同的模型族
token_patterns = {
    # Llama3 token IDs of "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    "llama3": [128009, 128006, 78191, 128007, 271],
    # Gemma2 token IDs of "<end_of_turn>\n<start_of_turn>model\n"
    "gemma2": [107, 108, 106, 2516, 108],
    # Qwen2 token IDs of "[b'print', b'("<', b'|', b'endo', b'ft', b'ext', b'|', b'>")']"
    "qwen2": [151644, 198, 151645, 198], # https://github.com/vitanova/Qwen2/blob/main/tokenization_note_zh.md
}

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


def find_token_for_gating(lst, model_family):
    # Find the last occurrence of a token_pattern in a list.  查找列表中最后一个出现的token_pattern
    token_pattern = token_patterns[model_family]
    token_pattern_len = len(token_pattern)
    search_end = len(lst)
    # print("lst = ", lst)
    # print("search_end = ", search_end)
    # print("token_pattern_len = ", token_pattern_len)
    for j in range(search_end - token_pattern_len, -1, -1):
        if lst[j : j + token_pattern_len] == token_pattern:
            return j
    raise ValueError("Token pattern not found in the list.")


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



#################### Initialize the argument parser to handle command-line inputs  初始化参数解析器以处理命令行输入
parser = ArgumentParser()
parser.add_argument(
    "--model_path",
    type=str,
    default="/mnt/prev_nas/virtual_agent/models/Qwen/Qwen2-VL-7B-Instruct",
    help="Path to the pre-trained model (HuggingFace path or local folder)",
)
parser.add_argument(
    "--model_family", type=str, default="qwen2", help="Model family (llama3 or gemma2)"
)
parser.add_argument(
    "--dataset_path",
    type=str,
    default="/mnt/prev_nas/virtual_agent/MBC/Reward_Model_csv_zeta/Reward_Model_preference_dataset.csv",
    help="Path to the dataset (HuggingFace path or local folder)",
)
parser.add_argument(
    "--source", default=None, type=str, help="Source filter for the dataset"
)
parser.add_argument(
    "--dataset_split", type=str, default="train", help="Dataset split to use"
)
parser.add_argument(
    "--n_shards",
    type=int,
    default=1,
    help="Total number of shards to divide the dataset into",
)
parser.add_argument(
    "--shard_idx", type=int, default=1, help="Index of the current shard"
)
parser.add_argument(
    "--device", type=int, default=0, help="CUDA device index to use for computation"
)
parser.add_argument(
    "--seq_len", type=int, default=8192, help="Maximum sequence length for input"
)
args = parser.parse_args()  # Parse the provided command-line arguments



# Load and prepare the dataset  加载并准备数据集
ds = datasets.load_dataset("csv", data_files=args.dataset_path, split=args.dataset_split)
if args.source is not None:
    ds = ds.filter(lambda x: x["source"] == args.source)
if args.n_shards > 1:
    ds = ds.shuffle(seed=0)
    ds = ds.shard(num_shards=args.n_shards, index=args.shard_idx - 1)


# Load the pre-trained model and tokenizer  加载预训练模型和tokenizer
device = f"cuda:{args.device}"
model = AutoModel.from_pretrained(
    args.model_path,
    torch_dtype=torch.bfloat16,  # Use bfloat16 precision for model weights to save memory  使用bfloat16精度作为模型权重以节省内存
    device_map=device,
    attn_implementation="flash_attention_2",  # Specify the attention implementation for efficiency  明确关注效率的实施
)
tokenizer = AutoTokenizer.from_pretrained(args.model_path)


# Initialize lists to store embeddings and prompt embeddings  初始化列表以存储embeddings以及prompt embeddings
embeddings = []
prompt_embeddings = []

# Process each example in the dataset  处理数据集中的每个示例
for example in tqdm(ds, desc="Examples"):
    chosen = example["chosen"]
    rejected = example["rejected"]

    if "prompt" in example:
        # Format the data with the standard chat template if prompt is available  如果有提示，请使用标准聊天模板格式化数据
        chosen = [
            {"role": "user", "content": example["prompt"]},
            {"role": "assistant", "content": chosen},
        ]
        rejected = [
            {"role": "user", "content": example["prompt"]},
            {"role": "assistant", "content": rejected},
        ]

    pair_embeddings = []
    pair_prompt_embeddings = []

    for iter_example in [chosen, rejected]:
        # Format the conversation messages using the tokenizer's chat template without tokenization  使用tokenizer的聊天模板格式化对话消息，无需tokenization
        if args.model_path.endswith("FsfairX-LLaMA3-RM-v0.1"):
            # Follows the demo code: https://huggingface.co/sfairXC/FsfairX-LLaMA3-RM-v0.1
            conv_formatted = tokenizer.apply_chat_template(
                iter_example, tokenize=False, add_generation_prompt=False
            ).replace(tokenizer.bos_token, "")
        else:
            conv_formatted = tokenizer.apply_chat_template(iter_example, tokenize=False)

        # Tokenize the formatted conversation and move tensors to the specified device  对格式化的对话进行Tokenize，并将张量移动到指定设备
        conv_tokenized = tokenizer(conv_formatted, return_tensors="pt").to(device)

        input_ids = conv_tokenized["input_ids"]

        # We only have one sequence so batch size is 1
        if input_ids.shape[1] > args.seq_len:
            continue

        with torch.no_grad():
            output = model(**conv_tokenized)
            last_hidden_state = output.last_hidden_state[0]

            # Find the position of the gating token and extract embeddings  找到门控标记的位置并提取嵌入
            gating_token_position = find_token_for_gating(
                input_ids[0].tolist(), args.model_family
            )
            prompt_embedding = last_hidden_state[gating_token_position].cpu()
            last_token_embedding = last_hidden_state[-1].cpu()

            pair_embeddings.append(last_token_embedding)
            pair_prompt_embeddings.append(prompt_embedding)

    # Only add the pair if both chosen and rejected embeddings were successfully computed  只有当所选和被拒绝的嵌入都成功计算时，才添加该对
    if len(pair_embeddings) == 2:
        embeddings.append(torch.stack(pair_embeddings))
        prompt_embeddings.append(torch.stack(pair_prompt_embeddings))

# Convert lists of embeddings to tensors  将嵌入列表转换为张量
embeddings = torch.stack(embeddings)
prompt_embeddings = torch.stack(prompt_embeddings)

