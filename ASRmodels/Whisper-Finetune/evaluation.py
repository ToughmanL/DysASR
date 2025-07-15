import argparse
import functools
import gc
import os
import platform
import time

import evaluate
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from utils.data_utils import DataCollatorSpeechSeq2SeqWithPadding, remove_punctuation, to_simple
from utils.reader import CustomDataset
from utils.utils import print_arguments, add_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg("test_data", type=str,  nargs='+', default=["dataset/test.json"], help="测试集的路径")
add_arg("model_path",  type=str, default="models/whisper-tiny-finetune", help="合并模型的路径，或者是huggingface上模型的名称")
add_arg("result_dir",  type=str, default="results/", help="计算出结果的保存路径")
add_arg("batch_size",  type=int, default=16,        help="评估的batch size")
add_arg("num_workers", type=int, default=8,         help="读取数据的线程数量")
add_arg("language",    type=str, default="Chinese", help="设置语言，可全称也可简写，如果为None则评估的是多语言")
add_arg("remove_pun",  type=bool, default=True,     help="是否移除标点符号")
add_arg("to_simple",   type=bool, default=True,     help="是否转为简体中文")
add_arg("timestamps",  type=bool, default=False,    help="评估时是否使用时间戳数据")
add_arg("min_audio_len",     type=float, default=0.5,  help="最小的音频长度，单位秒")
add_arg("max_audio_len",     type=float, default=30,   help="最大的音频长度，单位秒")
add_arg("local_files_only",  type=bool,  default=True, help="是否只在本地加载模型，不尝试下载")
add_arg("task",       type=str, default="transcribe", choices=['transcribe', 'translate'], help="模型的任务")
add_arg("metric",     type=str, default="cer",        choices=['cer', 'wer'],              help="评估方式")
args = parser.parse_args()
print_arguments(args)

# 判断模型路径是否合法
assert 'openai' == os.path.dirname(args.model_path) or os.path.exists(args.model_path), \
    f"模型文件{args.model_path}不存在，请检查是否已经成功合并模型，或者是否为huggingface存在模型"

# 如果是Windows，num_workers设置为0
if platform.system() == "Windows":
    args.num_workers = 0


def text_write(text_list, key_list, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        for key, text in zip(key_list, text_list):
            f.write(f"{key} {text}\n")
    print(f"保存结果到：{file_path}")

def result_write(result, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(str(result))
    print(f"保存结果到：{file_path}")


def get_evaluate(model, processor, test_data, args):
    # 获取测试数据
    test_dataset = CustomDataset(data_list_path=test_data,
                                 processor=processor,
                                 timestamps=args.timestamps,
                                 min_duration=args.min_audio_len,
                                 max_duration=args.max_audio_len)
    print(f"测试数据：{len(test_dataset)}")
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
    eval_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=data_collator)
    metric = evaluate.load(f'metrics/{args.metric}.py')
    label_text, decode_text, key_list = [], [], []

    
    # total_params = sum(p.numel() for p in model.parameters())
    
    # 开始评估
    for step, batch in enumerate(tqdm(eval_dataloader)):
        with torch.autocast(device_type="cuda"):
            with torch.no_grad():
                generated_tokens = (
                    model.generate(
                        input_features=batch["input_features"].cuda(),
                        decoder_input_ids=batch["labels"][:, :4].cuda(),
                        max_new_tokens=255).cpu().numpy())
                labels = batch["labels"].cpu().numpy()
                keys = batch["key"]
                labels = np.where(labels != -100, labels, processor.tokenizer.pad_token_id)
                # 将预测和实际的token转换为文本
                decoded_preds = processor.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
                decoded_labels = processor.tokenizer.batch_decode(labels, skip_special_tokens=True)
                # 删除标点符号
                if args.remove_pun:
                    decoded_preds = remove_punctuation(decoded_preds)
                    decoded_labels = remove_punctuation(decoded_labels)
                # 将繁体中文总成简体中文
                if args.to_simple:
                    decoded_preds = to_simple(decoded_preds)
                    decoded_labels = to_simple(decoded_labels)
                metric.add_batch(predictions=decoded_preds, references=decoded_labels)
                # 记录预测和实际的文本
                label_text.extend(decoded_labels)
                decode_text.extend(decoded_preds)
                key_list.extend(keys)
        # 删除计算的记录
        del generated_tokens, labels, batch, keys
        gc.collect()
        
    # 计算评估结果
    m = metric.compute()
    print(f"评估结果：{args.metric}={m}")
    return m, label_text, decode_text, key_list


def main():
    # 获取Whisper的数据处理器，这个包含了特征提取器、tokenizer
    processor = WhisperProcessor.from_pretrained(args.model_path,
                                                 language=args.language,
                                                 task=args.task,
                                                 no_timestamps=not args.timestamps,
                                                 local_files_only=args.local_files_only)
    # 获取模型
    model = WhisperForConditionalGeneration.from_pretrained(args.model_path,
                                                            device_map="auto",
                                                            local_files_only=args.local_files_only)
    model.generation_config.language = args.language.lower()
    model.eval()

    for test_data in args.test_data:
        m, label_text, decode_text, key_list = get_evaluate(model, processor, test_data, args)
        # 保存预测和实际的文本
        if os.path.exists(args.result_dir) is False:
            os.makedirs(args.result_dir)
        label_result_path = os.path.join(args.result_dir, f"{os.path.basename(test_data)}_decode.txt")
        decode_result_path = os.path.join(args.result_dir, f"{os.path.basename(test_data)}_label.txt")
        wer_result_path = os.path.join(args.result_dir, f"{os.path.basename(test_data)}_{args.metric}.txt")
        text_write(label_text, key_list, label_result_path)
        text_write(decode_text, key_list, decode_result_path)
        result_write(str(m), wer_result_path)



if __name__ == '__main__':
    main()
