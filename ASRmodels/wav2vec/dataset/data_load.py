# -*- encoding: utf-8 -*-
'''
File       :data_load.py
Description:
Date       :2025/02/27
Author     :lxk
Version    :1.0
Contact		:xk.liu@siat.ac.cn
License		:GPL
'''



import soundfile as sf
import torch
from datasets import Dataset
from transformers import Wav2Vec2Processor
import os
import json


processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")

# 音频处理函数
def process_audio(example):
    audio_input, sample_rate = sf.read(example['wav'])
    # 确保采样率与预训练模型一致
    if sample_rate != 16000:
        raise ValueError(f"Sample rate should be 16000, but got {sample_rate}")
    
    # 对音频进行处理
    input_values = processor(audio_input, sampling_rate=sample_rate, return_tensors="pt").input_values
    return {'input_values': input_values[0]}

# 处理数据集
processed_dataset = custom_dataset.map(process_audio, remove_columns=["wav"])

# 为目标文本进行处理
def process_text(example):
    with processor.as_target_processor():
        labels = processor(example['txt'], return_tensors="pt").input_ids
    return {'labels': labels[0]}

processed_dataset = processed_dataset.map(process_text, remove_columns=["txt"])


# 加载自定义数据集
def load_custom_dataset_from_json(json_file_path):
    data = []
    with open(json_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line.strip())
            data.append({
                'wav': entry['wav'],
                'txt': entry['txt']
            })
    return Dataset.from_list(data)

# 假设你的JSON文件路径如下
json_file_path = "path/to/your/data.json"

# 加载数据集
custom_dataset = load_custom_dataset_from_json(json_file_path)
