# -*- encoding: utf-8 -*-
'''
File       :dict_prep.py
Description:
Date       :2025/02/27
Author     :lxk
Version    :1.0
Contact		:xk.liu@siat.ac.cn
License		:GPL
'''


import collections

def generate_vocab(input_file, output_file, special_symbols=None):
    """
    生成字典文件，包含文本文件中的所有词语（按需添加特殊符号）。
    
    参数:
    input_file (str): 输入的文本文件路径，每行包含文本。
    output_file (str): 输出字典文件的路径。
    special_symbols (list): 要添加到字典的特殊符号（默认包含：<blank>, <unk>, <pad>, <bos>, <eos>）。
    
    返回:
    None
    """
    if special_symbols is None:
        special_symbols = ['<blank>', '<unk>', '<pad>', '<bos>', '<eos>']
    

    # 读取训练文本文件
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # 提取所有单词
    words = []
    for line in lines:
        words.extend(list(line.strip()))


    # 统计词频
    word_counts = collections.Counter(words)

    with open(output_file, 'w', encoding='utf-8') as f:
        for word, count in word_counts.items():
            f.write(f"{word} {count}\n")

    # 将字典保存到文件
    # with open(output_file, 'w', encoding='utf-8') as f:
    #     for idx, word in enumerate(vocab):
    #         f.write(f"{word} {idx}\n")
    


    print(f"字典已生成并保存为 {output_file}")

if __name__ == '__main__':

  # 示例使用
  input_file = 'data/zh.wrd'  # 输入文件路径
  output_dict_file = 'data/zh_dict.txt'  # 输出字典文件路径

  generate_vocab(input_file, output_dict_file)
