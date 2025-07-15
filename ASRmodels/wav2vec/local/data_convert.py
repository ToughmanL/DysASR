# -*- encoding: utf-8 -*-
'''
File       :data_convert.py
Description:
Date       :2025/02/27
Author     :lxk
Version    :1.0
Contact		:xk.liu@siat.ac.cn
License		:GPL
'''

import os
import json
# import torchaudio
from collections import Counter

import soundfile


def datalist2tsv(datalist_path, output_dir, datapart, lang):
  os.makedirs(output_dir, exist_ok=True)

  # 输出文件路径
  tsv_file = os.path.join(output_dir, f"{datapart}.tsv")
  wrd_file = os.path.join(output_dir, f"{datapart}.wrd")
  ltr_file = os.path.join(output_dir, f"{datapart}.ltr")

  # 读取 JSON 数据
  with open(datalist_path, "r", encoding="utf-8") as f:
      data = [json.loads(line.strip()) for line in f]

  # 生成 TSV 文件（第一行为音频目录）
  with open(tsv_file, "w", encoding="utf-8") as f_tsv:
      for item in data:
          wav_path = item["wav"]
          txt = item["txt"]
          # 如果中文的话插入空格
          if lang == 'ZH':
            txt = " ".join(list(txt))
          frames = soundfile.info(wav_path).frames
          f_tsv.write(f"{wav_path}\t{frames}\n")

  # 生成 WRD 和 LTR 文件
  char_counter = Counter()
  with open(wrd_file, "w", encoding="utf-8") as f_wrd, open(ltr_file, "w", encoding="utf-8") as f_ltr:
      for item in data:
          text = item["txt"]
          # 小写改为大写
          text = text.upper()
          f_wrd.write(text + "\n")  # 逐行写入完整句子

          # 逐字符转换（空格用 '|' 表示）
          ltr_text = " ".join(list(text.replace(" ", "|"))) + " |"
          f_ltr.write(ltr_text + "\n")

          # 更新字符统计
          char_counter.update(list(text.replace(" ", "|")) + ["|"])
  print(f"转换完成，生成文件保存在 {output_dir} {datapart}/")
  return char_counter

# 生成词汇表（字频降序）
def generate_vocab_file(char_counter, output_dir):
  dict_file = os.path.join(output_dir, "dict.ltr.txt")
  with open(dict_file, "w", encoding="utf-8") as f_dict:
    for char, freq in char_counter.most_common():
        f_dict.write(f"{char} {freq}\n")
  print(f"生成词汇表文件 {dict_file}")


if __name__ == "__main__":
  # /mnt/shareEEx/liuxiaokang/workspace/wenet/wenet-3.1.0/examples/cdsd/250224/data
  # /mnt/shareEEx/liuxiaokang/workspace/wenet/wenet-3.1.0/examples/msdm/250225/data
  # /mnt/shareEEx/liuxiaokang/workspace/wenet/wenet-3.1.0/examples/voicebank/250224/data
  # /mnt/shareEEx/liuxiaokang/workspace/wenet/wenet-3.1.0/examples/TROGO/250224/data
  # /mnt/shareEEx/liuxiaokang/workspace/wenet/wenet-3.1.0/examples/uaspeech/250225/data
  # dataset_dict = {'CDSD': 'cdsd/250224/data', 'MSDM': 'msdm/250225/data', 'VoiceBank': 'voicebank/250224/data', 'TROGO': 'TROGO/250224/data', 'UASpeech': 'uaspeech/250225/data', 'SpeechAccessibility':'SpeechAccessibility/conformer/data'}
  dataset_dict = {'SpeechAccessibility':'SpeechAccessibility/conformer/data'}
  ZH_list = ['CDSD', 'MSDM', 'VoiceBank']
  EN_list = ['TROGO', 'UASpeech', 'SpeechAccessibility']

  for key, value in dataset_dict.items():
    datalist_dir = f'/mnt/shareEEx/liuxiaokang/workspace/wenet/wenet-3.1.0/examples/{value}'
    output_dir = f'data/{key}'

    if key in ZH_list:
      lang = 'ZH'
    elif key in EN_list:
      lang = 'EN'

    train_datalist = os.path.join(datalist_dir, "train", 'data.raw.list')
    dev_datalist = os.path.join(datalist_dir, "dev", 'data.raw.list')
    test_datalist = os.path.join(datalist_dir, "test", 'data.raw.list')

    # 复制路径
    os.system(f"cp {train_datalist} {output_dir}/train.data.list")
    os.system(f"cp {dev_datalist} {output_dir}/dev.data.list")
    os.system(f"cp {test_datalist} {output_dir}/test.data.list")

    # _ = datalist2tsv(dev_datalist, output_dir, "dev", lang)
    # _ = datalist2tsv(test_datalist, output_dir, "test", lang)
    # char_counter = datalist2tsv(train_datalist, output_dir, "train", lang)
    # # generate_vocab_file(char_counter, output_dir)
    # print(f"数据集 {key} 转换完成！")
