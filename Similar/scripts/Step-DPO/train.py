import logging
import random
import sys

import torch
import transformers
from transformers import AutoModelForCausalLM, set_seed, AutoProcessor
from dataclasses import dataclass, field

import requests
from PIL import Image
from io import BytesIO

from alignment import (
    DataArguments,
    DPOConfig,
    H4ArgumentParser,
    ModelArguments,
    get_checkpoint,
    get_datasets,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
    get_tokenizer,
)
from stepdpo_trainer import StepDPOTrainer

from datasets import load_dataset, features


processor = AutoProcessor.from_pretrained("xxx", do_image_splitting=False)

def encode_image(image_content):
    return base64.b64encode(image_content).decode('utf-8')

def format(example):
    # Prepare the input for the chat template
    prompt = [
        {
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": example["prompt"] + "\n" + example['initial_reason_steps']}],
        },
    ]
    chosen = [
        {
            "role": "assistant",
            "content": [{"type": "text", "text": example["chosen"]}],
        },
    ]
    rejected = [
        {
            "role": "assistant",
            "content": [{"type": "text", "text": example["rejected"]}],
        },
    ]
    # Apply the chat template
    prompt = processor.apply_chat_template(prompt, tokenize=False)
    chosen = processor.apply_chat_template(chosen, tokenize=False)
    rejected = processor.apply_chat_template(rejected, tokenize=False)

    images = Image.open(BytesIO(requests.get(example["image_url"]).content))
    # Resize the image to ensure it fits within the maximum allowable
    # size of the processor to prevent OOM errors.
    max_size = processor.image_processor.size["longest_edge"] // 2
    images.thumbnail((max_size, max_size))

    return {"prompt": prompt, "images": images, "chosen": chosen, "rejected": rejected}



logger = logging.getLogger(__name__)

# 应用步骤级chat模板
def apply_step_wise_chat_template(
    example,
    tokenizer, 
    task, 
    prompt, 
    auto_insert_empty_system_msg: bool = True
):
    assert task in ["dpo"]
    if prompt == 'alpaca':
        prompt_input = (
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
        )
        prompt_no_input = (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Response:"
        )
    elif prompt == 'deepseek-math':
        prompt_input = None
        prompt_no_input = "User: {instruction}\nPlease reason step by step, and put your final answer within \\boxed{{}}.\n\nAssistant:"
    elif prompt == 'qwen2-boxed':
        prompt_input = None
        prompt_no_input = (
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            "<|im_start|>user\n{instruction}\nPlease reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

    text_chosen = example['chosen']
    text_rejected = example['rejected']

    if prompt == 'alpaca':
        if len(example['initial_reason_steps']) == 0:
            new_example = {
                'prompt': prompt_no_input.format(instruction=example['prompt']),
                'chosen': text_chosen,
                'rejected': text_rejected,
            }
        else:
            new_example = {
                'prompt': prompt_no_input.format(instruction=example['prompt']) + "\n" + example['initial_reason_steps'],
                'chosen': text_chosen,
                'rejected': text_rejected,
            }
    elif prompt == 'deepseek-math':
        if len(example['initial_reason_steps']) == 0:
            new_example = {
                'prompt': prompt_no_input.format(instruction=example['prompt']),
                'chosen': text_chosen,
                'rejected': text_rejected,
            }
        else:
            new_example = {
                'prompt': prompt_no_input.format(instruction=example['prompt']) + " " + example['initial_reason_steps'],
                'chosen': text_chosen,
                'rejected': text_rejected,
            }
    elif prompt == 'qwen2-boxed':
        if len(example['initial_reason_steps']) == 0:
            new_example = {
                'prompt': prompt_no_input.format(instruction=example['prompt']),
                'chosen': text_chosen,
                'rejected': text_rejected,
            }
        else:
            new_example = {
                'prompt': prompt_no_input.format(instruction=example['prompt']) + example['initial_reason_steps'],
                'chosen': text_chosen,
                'rejected': text_rejected,
            }

    return new_example

@dataclass
class StepDPOConfig(DPOConfig):
    data_path: str = field(default="xinlai/math-step-dpo-10K")
    prompt: str = field(default="alpaca")

# 主函数
def main():
    # 参数配置ss
    parser = H4ArgumentParser((ModelArguments, DataArguments, StepDPOConfig))
    model_args, data_args, training_args = parser.parse()

    #######
    # Setup 初始化
    #######
    # 设定logger
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary 总结当前进程信息
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Data parameters {data_args}")
    logger.info(f"Training/evaluation parameters {training_args}")

    # Check for last checkpoint 检查上一个checkpoint
    last_checkpoint = get_checkpoint(training_args)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    # Set seed for reproducibility 设定随机种子
    set_seed(training_args.seed)

    ###############
    # Load datasets 载入数据集
    ###############
    if ".json" in training_args.data_path:
        raw_datasets = load_dataset(
            "json",
            data_files=training_args.data_path.split("||"),
        )
    elif ".csv" in training_args.data_path:
        raw_datasets = load_dataset("csv", data_files=training_args.data_path)
    else:
        raw_datasets = load_dataset(training_args.data_path)

    logger.info(
        f"Training on the following splits: {[split + ' : ' + str(dset.num_rows) for split, dset in raw_datasets.items()]}"
    )
    column_names = list(raw_datasets["train"].features)
    print("column_names = ", column_names)

    #####################################
    # Load tokenizer and process datasets 载入tokenizer并处理数据集
    #####################################
    data_args.truncation_side = "left"  # Truncate from left to ensure we don't lose labels in final turn
    tokenizer = get_tokenizer(model_args, data_args)

    #####################
    # Apply chat template 应用chat模板
    #####################

    # 将chat模板原始数据集
    # raw_datasets = raw_datasets.map(
    #     apply_step_wise_chat_template,
    #     fn_kwargs={
    #         "tokenizer": tokenizer,
    #         "task": "dpo",
    #         "prompt": training_args.prompt,
    #         "auto_insert_empty_system_msg": data_args.auto_insert_empty_system_msg,
    #     },
    #     num_proc=data_args.preprocessing_num_workers,
    #     remove_columns=column_names,
    #     desc="Formatting comparisons with prompt template",
    # )
    raw_datasets = raw_datasets.map(
        format,
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=dataset.column_names,
        desc="Formatting comparisons with multimodal prompt template",
    ) # 1111111
    f = raw_datasets.features
    f["images"] = features.Sequence(features.Image(decode=True)) # to avoid bytes
    raw_datasets = raw_datasets.cast(f)


    # 记录训练集上的随机采样
    # Log a few random samples from the training set:
    for index in random.sample(range(len(raw_datasets["train"])), 3):
        logger.info(f"Prompt sample {index} of the raw training set:\n\n{raw_datasets['train'][index]['prompt']}")
        logger.info(f"Chosen sample {index} of the raw training set:\n\n{raw_datasets['train'][index]['chosen']}")
        logger.info(f"Rejected sample {index} of the raw training set:\n\n{raw_datasets['train'][index]['rejected']}")

    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    quantization_config = get_quantization_config(model_args) # 量化配置

    # 模型kw参数
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        use_flash_attention_2=model_args.use_flash_attention_2,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )

    # 模型和参照模型
    model = model_args.model_name_or_path
    ref_model = model
    ref_model_kwargs = model_kwargs
    print("model = ", model)

    if model_args.use_peft is True:
        ref_model = None
        ref_model_kwargs = None

    #########################
    # Instantiate DPO trainer 实例化dpo训练器
    #########################
    trainer = StepDPOTrainer(
        model,
        ref_model,
        model_init_kwargs=model_kwargs,
        ref_model_init_kwargs=ref_model_kwargs,
        args=training_args,
        beta=training_args.beta,
        train_dataset=raw_datasets["train"],
        eval_dataset=raw_datasets["test"] if "test" in raw_datasets.keys() else None,
        tokenizer=tokenizer,
        max_length=training_args.max_length,
        max_prompt_length=training_args.max_prompt_length,
        peft_config=get_peft_config(model_args),
        loss_type=training_args.loss_type,
    )

    ###############
    # Training loop 训练的loop
    ###############
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint

    train_result = trainer.train(resume_from_checkpoint=checkpoint) # 训练
    metrics = train_result.metrics
    metrics["train_samples"] = len(raw_datasets["train"])
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    logger.info("*** Training complete ***") # 完成训练

    ##################################
    # Save model and create model card 保存训练后的模型
    ##################################
    logger.info("*** Save model ***")
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

    # Save everything else on main process
    kwargs = {
        "finetuned_from": model_args.model_name_or_path,
        "dataset": [training_args.data_path],
        "dataset_tags": [training_args.data_path],
        "tags": ["alignment-handbook"],
    }
    if trainer.accelerator.is_main_process:
        trainer.create_model_card(**kwargs)
        # Restore k,v cache for fast inference
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)

    ##########
    # Evaluate 评估
    ##########
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(raw_datasets["test"])
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.push_to_hub is True:
        logger.info("Pushing to hub...")
        trainer.push_to_hub(**kwargs)

    logger.info("*** Training complete! ***")


if __name__ == "__main__":
    main()
