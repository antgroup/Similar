import os
import torch
import datasets
import numpy as np
from transformers import AutoTokenizer, AutoModel
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

# Define attributes (reward objectives)  定义属性
attributes = [
    "IP",
    "E",
    "TC",
    "TR",
    "C",
]

# Initialize the argument parser to handle command-line inputs  初始化参数parser
parser = ArgumentParser()
parser.add_argument(
    "--model_path",
    type=str,
    default="/mnt/prev_nas/virtual_agent/models/Qwen/Qwen2-VL-7B-Instruct",
    help="Path to the pre-trained model (HuggingFace path or local folder)",
)
parser.add_argument(
    "--dataset_path",
    type=str,
    default="/mnt/prev_nas/virtual_agent/MBC/Reward_Model_csv_zeta/Reward_Model_multi-objective_dataset.csv",
    help="Path to the dataset (HuggingFace path or local folder)",
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
args = parser.parse_args()  # Parse the provided command-line arguments


# Load the specified dataset and prepare it for processing  加载指定的数据集并准备处理
ds = datasets.load_dataset("csv", data_files=args.dataset_path)["train"]  # Load the training split of the dataset
ds = ds.shuffle(seed=0)  # Shuffle the dataset to ensure randomness
if args.n_shards > 1:
    ds = ds.shard(
        num_shards=args.n_shards, index=args.shard_idx - 1
    )  # Divide dataset into shards if needed

# Load the pre-trained model and tokenizer from the specified path  从指定路径加载预训练的模型和tokenizer
rm = AutoModel.from_pretrained(
    args.model_path,
    torch_dtype=torch.bfloat16,  # Use bfloat16 precision for model weights to save memory
    attn_implementation="flash_attention_2",  # Specify the attention implementation for efficiency
)

device = f"cuda:{args.device}"  # Define the CUDA device string
rm = rm.to(device)  # Move the model to the specified CUDA device
rm_tokenizer = AutoTokenizer.from_pretrained(
    args.model_path
)  # Load the tokenizer associated with the model

# Initialize lists to store embeddings and corresponding labels  初始化列表以存储嵌入和相应的标签
embeddings = []
labels = []

# Iterate over each example in the dataset with a progress bar  使用进度条迭代数据集中的每个示例
for example in tqdm(ds, desc="Processing dataset"):
    # Format the conversation messages using the tokenizer's chat template without tokenization  使用tokenizer的聊天模板格式化对话消息，无需tokenization
    # print("example = ", example)
    if args.model_path.endswith("FsfairX-LLaMA3-RM-v0.1"):
        # Follows the demo code: https://huggingface.co/sfairXC/FsfairX-LLaMA3-RM-v0.1
        conv_formatted = rm_tokenizer.apply_chat_template(
            example["messages"], tokenize=False, add_generation_prompt=False
        ).replace(rm_tokenizer.bos_token, "")
    else:
        conv_formatted = rm_tokenizer.apply_chat_template(
            example["messages"], tokenize=False
        )
    # Tokenize the formatted conversation and move tensors to the specified device  对格式化的对话进行tokenize，并将张量移动到指定设备
    conv_tokenized = rm_tokenizer(conv_formatted, return_tensors="pt").to(device)

    with torch.no_grad():
        output = rm(**conv_tokenized)  # Forward pass through the model
        # Extract the last hidden state of the last token and move it to CPU  提取最后一个令牌的最后一个隐藏状态并将其移动到CPU
        embeddings.append(output.last_hidden_state[0][-1].cpu())

    # Extract labels for the current example based on predefined attributes  根据预定义的属性提取当前示例的标签
    label = [example[attr] for attr in attributes]
    # Replace None values with NaN for consistent numerical processing  用NaN替换None值以进行一致的数值处理
    label = [np.nan if l is None else l for l in label]
    labels.append(label)  # Append the processed labels to the list  将处理后的标签附加到列表中

# Convert the list of labels to a NumPy array with float32 precision  将标签列表转换为精度为float32的NumPy数组
labels = np.array(labels, dtype=np.float32)
labels = torch.from_numpy(labels)  # Convert the NumPy array to a PyTorch tensor  将NumPy数组转换为PyTorch张量
embeddings = torch.stack(embeddings, dim=0)  # Stack all embeddings into a single tensor  将所有嵌入堆叠到一个张量中

# Define the path to save the embeddings and labels  定义保存embeddings和labels的路径
# HOME = os.path.expanduser("~")  # Get the home directory of the current user
# model_name = args.model_path.split("/")[-1]  # Extract the model name from the model path
# dataset_name = args.dataset_path.split("/")[-1]  # Extract the dataset name from the dataset path
# save_path = os.path.join(HOME, "data", "ArmoRM", "embeddings", model_name, dataset_name)  # Construct the save directory path
save_path = "/mnt/prev_nas/virtual_agent/MBC/RLHF-Reward-Modeling/checkpoints/checkpoints"
os.makedirs(save_path, exist_ok=True)  # Create the directory if it doesn't exist

# Save the embeddings and labels in a safetensors file with shard indexing
save_file(
    {"embeddings": embeddings, "labels": labels},
    f"{save_path}-{args.shard_idx:05d}-of-{args.n_shards:05d}.safetensors",
)

# Print a confirmation message with the path to the saved embeddings
print(
    f"Saved embeddings to {save_path}-{args.shard_idx:05d}-of-{args.n_shards:05d}.safetensors"
)
