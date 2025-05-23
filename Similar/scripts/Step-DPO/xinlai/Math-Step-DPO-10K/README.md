---
language:
- en
dataset_info:
  features:
  - name: dataset
    dtype: string
  - name: prompt
    dtype: string
  - name: initial_reason_steps
    dtype: string
  - name: chosen
    dtype: string
  - name: rejected
    dtype: string
  - name: full_chosen
    dtype: string
  - name: full_rejected
    dtype: string
  - name: answer
    dtype: string
  splits:
  - name: train
    num_bytes: 26528471
    num_examples: 10795
  download_size: 11985248
  dataset_size: 26528471
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
tags:
- dpo
---

# Step-DPO: Step-wise Preference Optimization for Long-chain Reasoning of LLMs

🖥️[Code](https://github.com/dvlab-research/Step-DPO) | 🤗[Data](https://huggingface.co/datasets/xinlai/Math-Step-DPO-10K) | 📄[Paper](https://arxiv.org/pdf/2406.18629)

This repo contains the **Math-Step-DPO-10K** dataset for our paper **Step-DPO: Step-wise Preference Optimization for Long-chain Reasoning of LLMs**, **Step-DPO** is a simple, effective, and data-efficient method for boosting the mathematical reasoning ability of LLMs. Notably, Step-DPO, when applied to Qwen2-72B-Instruct, achieves scores of **70.8%** and **94.0%** on the test sets of **MATH** and **GSM8K** without bells and wistles, respectively, surpassing a series of closed-source models, including GPT-4-1106, Claude-3-Opus, and Gemini-1.5-Pro.

**Math-Step-DPO-10K** is a high-quality step-wise preference dataset for mathematical reasoning.

![image/png](https://github.com/dvlab-research/Step-DPO/blob/main/imgs/coreidea.png)


## Contact

Please submit an issue [here](https://github.com/dvlab-research/Step-DPO) or send me an email [here](mailto:xinlai@cse.cuhk.edu.hk).
