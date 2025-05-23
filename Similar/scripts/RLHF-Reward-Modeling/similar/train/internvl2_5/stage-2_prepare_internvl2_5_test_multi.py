import os
import torch
import datasets
from transformers import AutoTokenizer, AutoModel, AutoConfig
from tqdm.auto import tqdm
from safetensors.torch import save_file
from argparse import ArgumentParser

import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from io import BytesIO
import requests as req

from internvl_conversation import get_conv_template

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
    "internvl2_5": [92542, 364, 92543, 525, 11353, 364],
}


def find_token_for_gating(lst, model_family):
    # Find the last occurrence of a token_pattern in a list.  查找列表中最后一个出现的token_pattern
    token_pattern = token_patterns[model_family]
    token_pattern_len = len(token_pattern)
    search_end = len(lst)
    # print("lst = ", lst)
    for j in range(search_end - token_pattern_len, -1, -1):
        if lst[j : j + token_pattern_len] == token_pattern:
            return j
    raise ValueError("Token pattern not found in the list.")


# Initialize the argument parser to handle command-line inputs  初始化参数解析器以处理命令行输入
parser = ArgumentParser()
parser.add_argument(
    "--model_path",
    type=str,
    default="/mnt/prev_nas/virtual_agent/models/OpenGVLab/InternVL2_5-8B", # https://huggingface.co/sfairXC/FsfairX-LLaMA3-RM-v0.1
    help="Path to the pre-trained model (HuggingFace path or local folder)",
)
parser.add_argument(
    "--model_family", type=str, default="internvl2_5", help="Model family (llama3 or gemma2)"
)
parser.add_argument(
    "--dataset_path",
    type=str,
    default="/mnt/prev_nas/virtual_agent/MBC/Reward_Model_csv_zeta/Reward_Model_preference_dataset_test_modify.csv", # https://huggingface.co/datasets/RLHFlow/UltraFeedback-preference-standard
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
# parser.add_argument(
#     "--device", type=int, default=0, help="CUDA device index to use for computation"
# )
parser.add_argument(
    "--seq_len", type=int, default=8192, help="Maximum sequence length for input"
)
args = parser.parse_args()  # Parse the provided command-line arguments


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


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
save_path = "/mnt/prev_nas/virtual_agent/MBC/RLHF-Reward-Modeling/checkpoints/gating_network_internvl2_5_embedding_test_modify"
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

# ds = ds.select(range(10))


# Load the pre-trained model and tokenizer  加载预训练模型和tokenizer
# device = f"cuda:{args.device}"
model = AutoModel.from_pretrained(
    args.model_path,
    torch_dtype=torch.bfloat16,  # Use bfloat16 precision for model weights to save memory  使用bfloat16精度作为模型权重以节省内存
    low_cpu_mem_usage=True,
    use_flash_attn=True,
    attn_implementation="flash_attention_2",  # Specify the attention implementation for efficiency  明确关注效率的实施
    trust_remote_code=True,
).eval().cuda()

if torch.cuda.device_count() > 1:
    from accelerate import dispatch_model
    from accelerate.utils import infer_auto_device_map, get_balanced_memory
    device_map = infer_auto_device_map(model, max_memory=get_balanced_memory(model))
    model = dispatch_model(model, device_map)
    # print('multi GPU predict => {}'.format(device_map))
else:
    model = model.cuda()
    # print("single GPU predict")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(
    args.model_path,
    trust_remote_code=True,
    use_fast=False,
)

# Initialize lists to store embeddings and prompt embeddings  初始化列表以存储embeddings以及prompt embeddings
embeddings = []
prompt_embeddings = []
IMG_START_TOKEN='<img>'
IMG_END_TOKEN='</img>'
IMG_CONTEXT_TOKEN='<IMG_CONTEXT>'

# Process each example in the dataset  处理数据集中的每个示例
for example in tqdm(ds, desc="Examples"):
    chosen = example["chosen"]
    rejected = example["rejected"]

    if "prompt" in example:
        # Format the data with the standard chat template if prompt is available  如果有prompt，请使用标准聊天模板格式化数据
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
        message = eval(iter_example) # 转为列表

        system_message = message[0]['content'][0]['text']
        user_message = message[1]['content'][0]['text']
        assistant_message = message[2]['content']
        image_url = message[1]['content'][1]['image']

        response = req.get(image_url)
        image = BytesIO(response.content)
        pixel_values = load_image(image, max_num=12).to(torch.bfloat16).cuda()
        # question = user_message + '\nAssistant Answer:\n' + assistant_message
        user_message = '<image>\n' + user_message

        num_patches_list = [pixel_values.shape[0]] if pixel_values is not None else []
        assert pixel_values is None or len(pixel_values) == sum(num_patches_list)

        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        model.img_context_token_id = img_context_token_id
        generation_config = dict(max_new_tokens=1024, do_sample=True)

        template = get_conv_template(model.template)
        template.system_message = model.system_message + '\n' + system_message  # system信息
        eos_token_id = tokenizer.convert_tokens_to_ids(template.sep.strip())

        # template.append_message(template.roles[0], example["messages"])
        template.append_message(template.roles[0], user_message)  # user
        template.append_message(template.roles[1], assistant_message)  # assistant
        query = template.get_prompt()
        for num_patches in num_patches_list:
            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * model.num_image_token * num_patches + IMG_END_TOKEN
            query = query.replace('<image>', image_tokens, 1)

        # Tokenize the formatted conversation and move tensors to the specified device  对格式化的对话进行tokenize，并将张量移动到指定设备
        model_inputs = tokenizer(query, return_tensors="pt")
        input_ids = model_inputs['input_ids'].to(device)
        attention_mask = model_inputs['attention_mask'].to(device)
        generation_config['eos_token_id'] = eos_token_id


        # We only have one sequence so batch size is 1
        if input_ids.shape[1] > args.seq_len:
            continue

        with torch.no_grad():
            # Find the position of the gating token and extract embeddings  找到门控标记的位置并提取嵌入
            gating_token_position = find_token_for_gating(
                input_ids[0].tolist(), args.model_family
            )

            visual_features = None
            assert model.img_context_token_id is not None
            if pixel_values is not None:
                if visual_features is not None:
                    vit_embeds = visual_features
                else:
                    vit_embeds = model.extract_feature(pixel_values)
                input_embeds = model.language_model.get_input_embeddings()(input_ids)
                B, N, C = input_embeds.shape
                input_embeds = input_embeds.reshape(B * N, C)

                input_ids = input_ids.reshape(B * N)
                selected = (input_ids == model.img_context_token_id)
                assert selected.sum() != 0
                input_embeds[selected] = vit_embeds.reshape(-1, C).to(input_embeds.device)

                input_embeds = input_embeds.reshape(B, N, C)
            else:
                input_embeds = model.language_model.get_input_embeddings()(input_ids)

            last_hidden_state = input_embeds[0]


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

print("len(embeddings) = ", len(embeddings))

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
