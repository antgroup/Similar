import os
import torch
import datasets
from transformers import AutoTokenizer, AutoModel, AutoConfig
from tqdm.auto import tqdm
from safetensors.torch import save_file
from argparse import ArgumentParser

# Set up CUDA optimizations for faster computation
torch.backends.cuda.matmul.allow_tf32 = (
    True  # Enable TensorFloat-32 matrix multiplication on CUDA
)
torch.backends.cudnn.allow_tf32 = (
    True  # Allow TensorFloat-32 in cuDNN for faster convolution operations
)

# Define token patterns for gating different model families  定义token模式，用于gating不同的模型族
token_patterns = {
    # Llama3 token IDs of "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    "llama3": [128009, 128006, 78191, 128007, 271],
    # Gemma2 token IDs of "<end_of_turn>\n<start_of_turn>model\n"
    "gemma2": [107, 108, 106, 2516, 108],
}


def find_token_for_gating(lst, model_family):
    # Find the last occurrence of a token_pattern in a list.  查找列表中最后一个出现的token_pattern
    token_pattern = token_patterns[model_family]
    token_pattern_len = len(token_pattern)
    search_end = len(lst)
    for j in range(search_end - token_pattern_len, -1, -1):
        if lst[j : j + token_pattern_len] == token_pattern:
            return j
    raise ValueError("Token pattern not found in the list.")


# Initialize the argument parser to handle command-line inputs  初始化参数解析器以处理命令行输入
parser = ArgumentParser()
parser.add_argument(
    "--model_path",
    type=str,
    default="/mnt/prev_nas/virtual_agent/MBC/ArmoRM/ArmoRM-Llama3-8B-v0.1", # https://huggingface.co/sfairXC/FsfairX-LLaMA3-RM-v0.1
    help="Path to the pre-trained model (HuggingFace path or local folder)",
)
parser.add_argument(
    "--model_family", type=str, default="llama3", help="Model family (llama3 or gemma2)"
)
parser.add_argument(
    "--dataset_path",
    type=str,
    default="/mnt/prev_nas/virtual_agent/MBC/Reward_Model_csv_zeta/Reward_Model_preference_dataset_train_prompt_simple.csv", # https://huggingface.co/datasets/RLHFlow/UltraFeedback-preference-standard
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


# Verify that the model family is passed correctly  验证模型族是否正确传递
# config = AutoConfig.from_pretrained(args.model_path)
# if args.model_family == "llama3":
#     assert config.model_type == "llama"
# elif args.model_family == "gemma2":
#     assert config.model_type == "gemma2"
# else:
#     raise ValueError(f"Model family {args.model_family} is not supported")

# Set up paths for saving embeddings  设置保存嵌入的路径
# HOME = os.path.expanduser("~")
# model_name = args.model_path.split("/")[-1]
# dataset_name = args.dataset_path.split("/")[-1]
# save_path = HOME + f"/data/ArmoRM/embeddings/{model_name}/{dataset_name}"
save_path = "/mnt/prev_nas/virtual_agent/MBC/RLHF-Reward-Modeling/checkpoints/gating_network_armo_embedding_train_prompt_1.7"
if args.source is not None:
    save_path += f"-{args.source}"
save_path += f"-{args.dataset_split}"

# Load and prepare the dataset  加载并准备数据集
ds = datasets.load_dataset("csv", data_files=args.dataset_path, split=args.dataset_split)
if args.source is not None:
    ds = ds.filter(lambda x: x["source"] == args.source)
if args.n_shards > 1:
    ds = ds.shuffle(seed=0)
    ds = ds.shard(num_shards=args.n_shards, index=args.shard_idx - 1)

# Load the pre-trained model and tokenizer  加载预训练模型和tokenizer
# device = f"cuda:{args.device}"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        iter_example = eval(iter_example)
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

# Prepare the directory for saving embeddings  准备用于保存嵌入的目录
# os.makedirs(os.path.dirname(save_path), exist_ok=True)
file_name = (
    f"{save_path}-{args.shard_idx:05d}-of-{args.n_shards:05d}"
    if args.n_shards > 1
    else save_path
)

# Save the embeddings and prompt embeddings using safetensors 使用safetensor保存嵌入并提示嵌入``
save_file(
    {"embeddings": embeddings, "prompt_embeddings": prompt_embeddings},
    f"{save_path}.safetensors",
)

# Print a confirmation message with the path to the saved embeddings
print(f"Saved embeddings to {save_path}.safetensors")
