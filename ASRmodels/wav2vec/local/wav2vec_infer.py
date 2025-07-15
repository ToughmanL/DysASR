# -*- encoding: utf-8 -*-
'''
File     :wav2vec_infer.py
Description:
Date     :2024/11/04
Author   :lxk
Version  :1.0
Contact		:xk.liu@siat.ac.cn
License		:GPL
'''

# !pip install transformers
# !pip install datasets
import re
import os
import shutil
import numpy as np
import json
import librosa
import torch
import random
import concurrent.futures
import torchaudio
# from datasets import load_dataset
from transformers import Wav2Vec2ForCTC, Wav2Vec2Config, Wav2Vec2Processor
from cer_wer import compute_cer_wer


# 设置随机种子（更改种子值以获得不同的随机选择）
seed = 42
random.seed(seed)

# 获取GPU
def get_device(id=None):
  if id == None:
    num_gpus = torch.cuda.device_count()  # 获取可用 GPU 数量
    gpu_id = random.randint(0, num_gpus - 1)  # 随机选择一个 GPU
    device = torch.device(f"cuda:{gpu_id}")
    print(f"Using GPU: cuda:{gpu_id}")
  if id < 0:
    return torch.device("cpu")
  else:
    device = torch.device(f"cuda:{id}")
  return device


# 复制.json配置文件
def copy_json_files(src_dir, det_dir):
  # 确保目标目录存在
  if not os.path.exists(det_dir):
    os.makedirs(det_dir)

  # 遍历源目录中的所有文件
  for filename in os.listdir(src_dir):
    # 检查文件是否是.json文件
    if filename.endswith('.json'):
      src_file = os.path.join(src_dir, filename)
      det_file = os.path.join(det_dir, filename)

      # 检查目标目录中是否已经存在该文件
      if not os.path.exists(det_file):
        # 复制文件
        shutil.copy2(src_file, det_file)
        print(f"Copied {filename} to {det_dir}")
      else:
        print(f"{filename} already exists in {det_dir}, skipping.")

# 查找最新的模型文件夹
def find_model_path(model_dir):
  # 初始化最大编号和对应的文件夹路径
  max_num = -1
  max_folder = None

  # 遍历目标目录下的所有文件夹
  for folder in os.listdir(model_dir):
    folder_path = os.path.join(model_dir, folder)
    if folder == "best_model":
      max_folder = folder_path
      break
    # 检查是否为文件夹
    if os.path.isdir(folder_path):
      # 使用正则表达式匹配以 model_ep{num}_ 开头的文件夹
      match = re.match(r'model_ep(\d+)_', folder)
      if match:
        # 提取数字编号
        num = int(match.group(1))
        # 如果当前编号大于最大编号，则更新最大编号和对应的文件夹路径
        if num > max_num:
          max_num = num
          max_folder = folder_path
  # 文件夹上一级文件夹
  next_folder = os.path.dirname(max_folder)
  copy_json_files(next_folder, max_folder)
  return max_folder

# 读取数据列表
def read_datalist(datalist):
  samples = []
  with open(datalist, 'r') as f:
    for line in f:
      line = line.strip('\n')
      sample = json.loads(line)
      samples.append(sample)
  return samples

# 重采样
def resample(audio_input, sample_rate, target_sample_rate):
  if sample_rate != target_sample_rate:
    audio_input = librosa.resample(audio_input, sample_rate, target_sample_rate)
  return audio_input, target_sample_rate


# 读取语音文本
def read_audio(wav_path):
  # audio_input, sample_rate =  librosa.load(wav_path, sr=None)
  # if sample_rate != 16000:
  #   audio_input, sample_rate = resample(audio_input, sample_rate, 16000)
  audio_input, sample_rate = torchaudio.load(wav_path, normalize=True)
  audio_input = audio_input.mean(dim=0, keepdim=True)
  if sample_rate != 16000:
    resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate,
                new_freq=16000,
                resampling_method="sinc_interpolation",  # 或"kaiser_window"
                lowpass_filter_width=6,
                rolloff=0.99,
                dtype=torch.float32
            )
    audio_input = resampler(audio_input)
  return audio_input.squeeze(0).numpy(), 16000

# 推理类
class Inference():
  def __init__(self, model_path, device):
    self.device = device
    self.model_path = model_path
    self.processor = Wav2Vec2Processor.from_pretrained(self.model_path)
    self.config = Wav2Vec2Config.from_pretrained(self.model_path)
    model = Wav2Vec2ForCTC.from_pretrained(self.model_path, config=self.config ,ignore_mismatched_sizes=True)
    self.model = model.to(device)
    self.model.eval()

  def inference(self, audio_input, sample_rate):
    input_values = self.processor(audio_input, sampling_rate=sample_rate, return_tensors="pt").input_values.to(self.device)
    with torch.no_grad(), torch.cuda.amp.autocast():
      logits = self.model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = self.processor.decode(predicted_ids[0])
    return transcription

# 解码数据列表
def decode_list(infer, input_data, output_file):
  samples = read_datalist(input_data)
  result_text = []
  for idx, sample in enumerate(samples):
    audio_input, sample_rate = read_audio(sample['wav'])
    key = sample['key']
    if 'start' in sample and 'end' in sample:
      start_frame = int(sample['start'] * sample_rate)
      end_frame = int(sample['end'] * sample_rate)
      sample_data = audio_input[start_frame:end_frame]
    else:
      sample_data = audio_input
    try:
      transcription = infer.inference(sample_data, sample_rate)
    except:
      print("Decode error:{}".format(key))
      continue
    print(key, transcription)
    result_text.append({"key":key, "pred":transcription, "ref":sample['txt']})
  
  with open(output_file, 'w') as f:
    for sample in result_text:
      line = sample['key'] + " " + sample['pred'] + "\n"
      f.write(line)
  return result_text
  print("Done!")


# 推理和计算WER
def process_dataset(dataset, out_dir, test_list, model_path=None, device=None):
  device = get_device(device)
  if model_path is None:
    model_path = find_model_path(f"models/{dataset}")
  infer = Inference(model_path, device)
  for test in test_list:
    input_data = f"data/{dataset}/{test}.data.list"
    output_file = f"{out_dir}/{dataset}/{test}.wav2vec2.text"
    if not os.path.exists(f"{out_dir}/{dataset}"):
      os.makedirs(f"{out_dir}/{dataset}")
    print(f"input_data: {input_data}, output_file: {output_file}, model_path: {model_path}")
    result_text = decode_list(infer, input_data, output_file)
    if dataset in ['TROGO', 'UASpeech', 'SpeechAccessibility']:
      wer_info = compute_cer_wer(result_text, cer_wer='wer')
    else:
      wer_info = compute_cer_wer(result_text, cer_wer='cer')
    print(dataset, dataset, wer_info)
    wer_path = f"{out_dir}/{dataset}/{test}.wer"
    with open(wer_path, 'w') as f:
      f.write(json.dumps(wer_info, indent=4))


if __name__ == '__main__':
  # CDSD  MSDM  TROGO  UASpeech  VoiceBank
  dataset_list = ['CDSD', 'MSDM', 'TROGO', 'UASpeech', 'VoiceBank', 'SpeechAccessibility']
  dataset = 'SpeechAccessibility'
  test_list = ['dev', 'test']
  out_dir = "results/finetune" # 输出目录,有无fintune "results/finetune" "results/nofinetune"

  if "nofinetune" not in out_dir:
    pretrained_model_dir = None
    # 测试每个数据集, 使用finetune模型
    process_dataset(dataset, out_dir, test_list, None, 1)
  else:
    if dataset in ['CDSD', 'MSDM', 'VoiceBank']:
      pretrained_model_dir = "/mnt/shareEEx/liuxiaokang/data/pretrainmodel/wav2vec2-large-xlsr-53-chinese-zh-cn-gpt"
    elif dataset in ['TROGO', 'UASpeech', 'SpeechAccessibility']:
      pretrained_model_dir = "/mnt/shareEEx/liuxiaokang/data/pretrainmodel/wav2vec2-large-xlsr-53-english" # 英文
    
    # 测试每个数据集, 使用预训练模型
    print("dataset:{}".format(dataset), "pretrained_model_dir:{}".format(pretrained_model_dir))
    process_dataset(dataset, out_dir, test_list, pretrained_model_dir, 1)


  # 使用ThreadPoolExecutor来并行处理每个数据集
  # with concurrent.futures.ThreadPoolExecutor() as executor:
  #   futures = []
  #   for dataset in dataset_list:
  #     pretrained_model_dir = f"models/{dataset}"
  #     futures.append(executor.submit(process_dataset, dataset, out_dir, test_list, pretrained_model_dir, 1))
  #   # 等待所有任务完成
  #   for future in concurrent.futures.as_completed(futures):
  #     try:
  #       future.results()
  #     except Exception as e:
  #       print(f"An error occurred: {e}")
  print("Done!")
