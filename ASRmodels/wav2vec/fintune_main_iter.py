# -*- encoding: utf-8 -*-
'''
File       :fintune_main.py
Description:
Date       :2025/02/27
Author     :lxk
Version    :1.0
Contact		:xk.liu@siat.ac.cn
License		:GPL
'''
import os
import re
import json
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Config, Wav2Vec2Processor
from transformers import HubertForCTC, HubertConfig, AutoProcessor
from torch.utils.data import IterableDataset, DataLoader
from datasets import Dataset
from transformers import AdamW
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from torch.nn.utils.rnn import pad_sequence
import torchaudio

from tqdm import tqdm
import pyarrow as pa

# 删除字符串中除中文、英文、数字之外的所有符号，并将英文字符转换为大写
def remove_special_characters(text):
    # 保留中文、英文字母、数字和空格
    text = ''.join(filter(lambda x: x.isalnum() or x == ' ' or '\u4e00' <= x <= '\u9fa5', text))
    # 合并多个空格为一个空格
    text = re.sub(r'\s+', ' ', text)
    # 转换为大写
    # text = text.upper()
    # 转换为小写
    text = text.lower()
    return text


# 数据长度过滤
def filter_data_length(data_list, duration=25, text_len=1):
    samples = []
    with open(data_list, 'r', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line.strip())
            audio_path = entry['wav']
            text = entry['txt']
            if 'duration' not in entry:
                duration_sec = 10
            else:
                duration_sec = entry['duration']  # 音频时长，单位为秒
            new_text = remove_special_characters(text)

            if len(new_text) < text_len:
                continue
            else:
                entry['txt'] = new_text  # 更新文本为处理后的文本

            if os.path.exists(audio_path):
                if duration_sec <= duration:
                    samples.append(entry)
    return samples


# Dataset
class AudioTextIterableDataset(IterableDataset):
    def __init__(self, data_list, processor):
        self.data_list = data_list  # 直接接收Python列表
        self.processor = processor
        # 预加载resampler避免重复创建
        self.resamplers = {}  # 不同采样率使用不同的resampler
    
    def _get_resampler(self, orig_freq):
        if orig_freq not in self.resamplers:
            self.resamplers[orig_freq] = torchaudio.transforms.Resample(
                orig_freq=orig_freq,
                new_freq=16000)
        return self.resamplers[orig_freq]

    def _load_audio(self, path):
        waveform, sample_rate = torchaudio.load(path, normalize=True)
        waveform = waveform.mean(dim=0, keepdim=True)  # 转为单声道
        if sample_rate != 16000:
            resampler = self._get_resampler(sample_rate)
            waveform = resampler(waveform)
        # 确保音频数据是float32类型
        waveform = waveform.to(torch.float32)
        audio_numpy = waveform.squeeze().numpy().astype('float32')
        # 确保最终是1D数组 (如果是单声道) 或 2D数组 (如果是立体声)
        if audio_numpy.ndim > 1:
            audio_numpy = np.squeeze(audio_numpy)
        return audio_numpy

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # 单进程
            iter_data = self.data_list
        else:  # 多进程分片
            iter_data = self.data_list[worker_info.id :: worker_info.num_workers]
        
        for entry in iter_data:
            audio = self._load_audio(entry['wav'])
            text = entry['txt']
            key = entry['key']
            # 先处理文本（CPU操作）
            with self.processor.as_target_processor():
                labels = self.processor(text, return_tensors="pt").input_ids[0]

            input_values = self.processor(
                audio, 
                sampling_rate=16000, 
                return_tensors="pt"
            ).input_values[0]

            yield {
                'input_values': input_values,
                'labels': labels
            }


# 数据加载器
def collate_fn(batch):
    input_values = pad_sequence([torch.tensor(item['input_values']) for item in batch], batch_first=True, padding_value=0)
    labels = pad_sequence([torch.tensor(item['labels']) for item in batch], batch_first=True, padding_value=-100)  # -100 is typically used for ignored tokens in the loss calculation
    return {'input_values': input_values, 'labels': labels}


# 验证
def evaluate(model, dataloader):
    model.eval()
    total_loss, total_samples  = 0, 0
    with torch.no_grad(), torch.cuda.amp.autocast():
        for batch in dataloader:
            inputs = batch['input_values'].to(model.device, non_blocking=True)
            labels = batch['labels'].to(model.device, non_blocking=True)
            outputs = model(inputs, labels=labels)
            total_loss += outputs.loss.item() * inputs.size(0)
            total_samples  += inputs.size(0)
    return total_loss / total_samples 


# 训练函数
def train(model, processor, train_dataloader, dev_dataloader, optimizer, scheduler, scaler, num_epochs, new_model_dir, accumulation_steps=1):
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        model.train()
        train_total_loss, val_total_loss = 0, 0
        train_total_samples = 0

        optimizer.zero_grad()
        for batch in train_dataloader:
            input_values = batch['input_values'].to(model.device)
            labels = batch['labels'].to(model.device)
            # 计算损失
            with torch.cuda.amp.autocast():
                outputs = model(input_values, labels=labels)
                loss = outputs.loss / accumulation_steps

            train_total_loss += loss.item() * accumulation_steps
            scaler.scale(loss).backward()
            train_total_samples += 1

            # 累积到设定步数后更新参数
            if train_total_samples % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        
        # 验证集
        val_total_loss = evaluate(model, dev_dataloader)
        scheduler.step(val_total_loss)

        if val_total_loss < best_val_loss:
            best_val_loss = val_total_loss
            model.save_pretrained(os.path.join(new_model_dir, "best_model"))
            processor.save_pretrained(os.path.join(new_model_dir, "best_model"))

        # 打印epoch结果
        print(f"\nEpoch {epoch+1}/{num_epochs} - "
              f"Train Loss: {train_total_loss / train_total_samples:.4f} - "
              f"Val Loss: {val_total_loss:.4f} - "
              f"LR: {optimizer.param_groups[0]['lr']:.2e}")

        # 定期保存检查点
        if (epoch + 1) % 5 == 0 or epoch == num_epochs - 1:
            model_save_path = os.path.join(new_model_dir, f"model_ep{epoch+1}.pt")
            model.save_pretrained(model_save_path)
            processor.save_pretrained(model_save_path)
            print(f"Model saved to {model_save_path}")

def infer(model, processor, dataloader):
    keys, transcriptions, refs = [], [], []
    # 推理
    model.eval()
    for batch in tqdm(dataloader):
        input_values = batch['input_values'].to(model.device, non_blocking=True)
        with torch.no_grad(), torch.cuda.amp.autocast():
            logits = model(input_values).logits
        batch_key = batch['keys']
        batch_labels = batch['txt']
        predicted_ids = torch.argmax(logits, dim=-1)
        batch_transcription = processor.decode(predicted_ids[0])
        kyes.extend(batch_key)
        transcriptions.extend(batch_transcription)
        refs.extend(batch_labels)
    
    result_text = []
    for key, pred, ref in zip(keys, transcriptions, refs):
        result_text.append({"key": key, "pred": pred, "ref": ref})
    return result_text

# 推理
def main_infer(data_dir, pretrained_model_dir, output_dir, device, batch_size=8):
    test_data_list = os.path.join(data_dir, "test.data.list")
    dev_data_list = os.path.join(data_dir, "dev.data.list")
    test_data_list = filter_data_length(test_data_list, duration=25, text_len=1)
    dev_data_list = filter_data_length(dev_data_list, duration=25, text_len=1)

    # 加载处理器
    processor = Wav2Vec2Processor.from_pretrained(pretrained_model_dir)

    test_dataset = AudioTextIterableDataset(test_data_list, processor)
    dev_dataset = AudioTextIterableDataset(dev_data_list, processor)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, pin_memory=True,persistent_workers=True, prefetch_factor=4, shuffle=False, collate_fn=collate_fn, num_workers=8)
    # test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn, num_workers=0)
    dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, pin_memory=True,persistent_workers=True, prefetch_factor=4, shuffle=False, collate_fn=collate_fn, num_workers=8)

    # 加载模型
    config = Wav2Vec2Config.from_pretrained(pretrained_model_dir)
    model = Wav2Vec2ForCTC.from_pretrained(pretrained_model_dir, config=config)
    model = model.to(device)

    test_result_text = infer(model, processor, test_dataloader)
    dev_result_text = infer(model, processor, dev_dataloader)

    # 保存结果
    for dataset in ["test", "dev"]:
        output_file = os.path.join(output_dir, f"{dataset}.wav2vec2.text")
        with open(output_file, 'w') as f:
            for sample in (test_result_text if dataset == "test" else dev_result_text):
                line = sample['key'] + " " + sample['pred'] + "\n"
                f.write(line)
        print(f"Results saved to {output_file}")


# 训练主函数
def main_train(model_name, data_dir, pretrained_model_dir, new_model_dir, device, batch_size=8, accumulation_steps=2, num_epochs=10):
    # 加载数据集
    train_data_list = os.path.join(data_dir, "train.data.list")
    dev_data_list = os.path.join(data_dir, "dev.data.list")

    # 数据长度过滤
    dev_data_list = filter_data_length(dev_data_list, duration=25, text_len=1)
    train_data_list = filter_data_length(train_data_list, duration=25, text_len=1)

    # 加载处理器
    if model_name == "wav2vec":
        processor = Wav2Vec2Processor.from_pretrained(pretrained_model_dir)
    elif model_name == "hubert":
        processor = AutoProcessor.from_pretrained(pretrained_model_dir)

    # 加载数据集
    train_dataset = AudioTextIterableDataset(train_data_list, processor)
    dev_dataset = AudioTextIterableDataset(dev_data_list, processor)

    # data loader
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, pin_memory=True,persistent_workers=True, prefetch_factor=4, shuffle=False, collate_fn=collate_fn, num_workers=8)
    dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, pin_memory=True,persistent_workers=True, prefetch_factor=4, shuffle=False, collate_fn=collate_fn, num_workers=8)

    # 模型
    if model_name == "wav2vec":
        config = Wav2Vec2Config.from_pretrained(pretrained_model_dir)
        model = Wav2Vec2ForCTC.from_pretrained(pretrained_model_dir, config=config)
    elif model_name == "hubert":
        config = HubertConfig.from_pretrained(pretrained_model_dir)
        model = HubertForCTC.from_pretrained(pretrained_model_dir, config=config)
    model = model.to(device)

    # 优化器, 学习率调度器, 混合精度
    optimizer = AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    scaler = torch.cuda.amp.GradScaler(enabled=True) # 混合精度优化

    # 开始训练
    train(model, processor, train_dataloader, dev_dataloader, optimizer, scheduler, scaler, num_epochs, new_model_dir, accumulation_steps)



# 主函数
if __name__ == "__main__":
    model_name = "wav2vec2"  # wav2vec2 or hubert
    data_set = "SpeechAccessibility" # CDSD  MSDM  VoiceBank TROGO  UASpeech SpeechAccessibility
    data_dir = f"data/{data_set}"

    pretrained_models = {'wav2vec:'{'en':"/mnt/shareEEx/liuxiaokang/data/pretrainmodel/wav2vec2-large-xlsr-53-english", 'zh':"/mnt/shareEEx/liuxiaokang/data/pretrainmodel/wav2vec2-large-xlsr-53-chinese-zh-cn-gpt"}, 'hubert':{'en':"/mnt/shareEEx/liuxiaokang/data/pretrainmodel/hubert-large-1160k", 'zh':"/mnt/shareEEx/liuxiaokang/data/pretrainmodel/hubert-large/chinese-hubert-large"}}

    lang_dict = {'CDSD': 'zh', 'MSDM': 'zh', 'VoiceBank': 'zh', 'TROGO': 'en', 'UASpeech': 'en', 'SpeechAccessibility': 'en'}

    pretrained_model_dir = pretrained_models[model_name][lang_dict[data_set]]
    new_model_dir = f"models/{model_name}/{data_set}"
    result_dir = f"results/finetune/{model_name}/{data_set}"
    if os.path.exists(new_model_dir) is False:
        os.makedirs(new_model_dir)

    gpu = 1
    device = torch.device(f"cuda:{gpu}")
    main_train(model_name, data_dir, pretrained_model_dir, new_model_dir, device, batch_size=4, accumulation_steps=4, num_epochs=100)

    # finetune_model_dir = os.path.join(new_model_dir, "best_model")
    # main_infer(data_dir, finetune_model_dir, result_dir, device, batch_size=4)

