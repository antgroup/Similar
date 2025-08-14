import os
from xmlrpc.client import Boolean

import torch
import datasets
import numpy as np
from transformers import AutoTokenizer, AutoModel
from tqdm.auto import tqdm
from safetensors.torch import save_file
from argparse import ArgumentParser

import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from io import BytesIO
import requests as req
from PIL import Image

from internvl_conversation import get_conv_template

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
    default="/mnt/prev_nas/virtual_agent/models/OpenGVLab/InternVL2_5-8B",
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
    # image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values



# Load the specified dataset and prepare it for processing  加载指定的数据集并准备处理
ds = datasets.load_dataset("csv", data_files=args.dataset_path)["train"]  # Load the training split of the dataset
# ds = ds.shuffle(seed=0).select(range(10))  # Shuffle the dataset to ensure randomness
if args.n_shards > 1:
    ds = ds.shard(
        num_shards=args.n_shards, index=args.shard_idx - 1
    )  # Divide dataset into shards if needed

# Load the pre-trained model and tokenizer from the specified path  从指定路径加载预训练的模型和tokenizer
rm = AutoModel.from_pretrained(
    args.model_path,
    torch_dtype=torch.bfloat16,  # Use bfloat16 precision for model weights to save memory
    low_cpu_mem_usage=True,
    use_flash_attn=True,
    attn_implementation="flash_attention_2",  # Specify the attention implementation for efficiency
    trust_remote_code=True,
).eval().cuda()

device = f"cuda:{args.device}"  # Define the CUDA device string
rm = rm.to(device)  # Move the model to the specified CUDA device
rm_tokenizer = AutoTokenizer.from_pretrained(
    args.model_path,
    trust_remote_code=True,
    use_fast=False,
)  # Load the tokenizer associated with the model

# Initialize lists to store embeddings and corresponding labels  初始化列表以存储嵌入和相应的标签
embeddings = []
labels = []
IMG_START_TOKEN='<img>'
IMG_END_TOKEN='</img>'
IMG_CONTEXT_TOKEN='<IMG_CONTEXT>'

# Iterate over each example in the dataset with a progress bar  使用进度条迭代数据集中的每个示例
for example in tqdm(ds, desc="Processing dataset"):
    # Format the conversation messages using the tokenizer's chat template without tokenization  使用tokenizer的聊天模板格式化对话消息，无需tokenization
    # print("example = ", example)
    message = eval(example["messages"])
    system_message = message[0]['content'][0]['text']
    user_message = message[1]['content'][0]['text']
    assistant_message = message[2]['content']
    image_url = message[1]['content'][1]['image']

    response = req.get(image_url)
    image = Image.open(BytesIO(response.content)).convert('RGB')
    image.thumbnail((512, 512))
    pixel_values = load_image(image, max_num=12).to(torch.bfloat16).cuda()
    # question = user_message + '\nAssistant Answer:\n' + assistant_message
    user_message = '<image>\n' + user_message


    num_patches_list = [pixel_values.shape[0]] if pixel_values is not None else []
    assert pixel_values is None or len(pixel_values) == sum(num_patches_list)

    img_context_token_id = rm_tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
    rm.img_context_token_id = img_context_token_id
    generation_config = dict(max_new_tokens=1024, do_sample=True)

    template = get_conv_template(rm.template)
    template.system_message = rm.system_message + '\n' + system_message # system信息
    eos_token_id = rm_tokenizer.convert_tokens_to_ids(template.sep.strip())

    # template.append_message(template.roles[0], example["messages"])
    template.append_message(template.roles[0], user_message) # user
    template.append_message(template.roles[1], assistant_message) # assistant
    query = template.get_prompt()
    for num_patches in num_patches_list:
        image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * rm.num_image_token * num_patches + IMG_END_TOKEN
        query = query.replace('<image>', image_tokens, 1)

    # Tokenize the formatted conversation and move tensors to the specified device  对格式化的对话进行tokenize，并将张量移动到指定设备
    model_inputs = rm_tokenizer(query, return_tensors="pt")
    input_ids = model_inputs['input_ids'].to(device)
    attention_mask = model_inputs['attention_mask'].to(device)
    generation_config['eos_token_id'] = eos_token_id

    with torch.no_grad():

        # output = rm.generate(
        #     pixel_values=pixel_values,
        #     input_ids=input_ids,
        #     attention_mask=attention_mask,
        #     **generation_config
        # )
        # print("output = ", output)
        # response = rm_tokenizer.batch_decode(output, skip_special_tokens=True)[0]
        # response = response.split(template.sep.strip())[0].strip()
        # print("\nresponse = \n", response)

        visual_features = None

        assert rm.img_context_token_id is not None
        if pixel_values is not None:
            if visual_features is not None:
                vit_embeds = visual_features
            else:
                vit_embeds = rm.extract_feature(pixel_values)
            input_embeds = rm.language_model.get_input_embeddings()(input_ids)
            B, N, C = input_embeds.shape
            input_embeds = input_embeds.reshape(B * N, C)

            input_ids = input_ids.reshape(B * N)
            selected = (input_ids == rm.img_context_token_id)
            assert selected.sum() != 0
            input_embeds[selected] = vit_embeds.reshape(-1, C).to(input_embeds.device)

            input_embeds = input_embeds.reshape(B, N, C)
        else:
            input_embeds = rm.language_model.get_input_embeddings()(input_ids)

        input_embeds = rm.language_model.model.norm(input_embeds)

        # print("\ninput_embeds = \n", input_embeds)
        # Extract the last hidden state of the last token and move it to CPU  提取最后一个token的最后一个隐藏状态并将其移动到CPU
        # embeddings.append(output.last_hidden_state[0][-1].cpu())
        embeddings.append(input_embeds[0][-1].cpu())

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
save_path = "/mnt/prev_nas/virtual_agent/MBC/RLHF-Reward-Modeling/checkpoints/regression_internvl2_5_embedding_1.5"
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
