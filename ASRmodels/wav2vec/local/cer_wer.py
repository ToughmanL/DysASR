# -*- encoding: utf-8 -*-
'''
File       :cer_wer.py
Description:
Date       :2025/03/01
Author     :lxk
Version    :1.0
Contact		:xk.liu@siat.ac.cn
License		:GPL
'''


import re
import os
import jiwer
from jiwer import cer, wer
import jiwer.transforms as tr
from typing import List

# 删除字符串中除中文、英文、数字之外的所有符号，并将英文字符转换为大写
def remove_special_characters(text):
    # 保留中文、英文字母、数字和空格
    text = ''.join(filter(lambda x: x.isalnum() or x == ' ' or '\u4e00' <= x <= '\u9fa5', text))
    # 合并多个空格为一个空格
    text = re.sub(r'\s+', ' ', text)
    # 转换为大写
    text = text.upper()
    return text


def compute_cer_wer(result_text, cer_wer='cer'):
    total = 0
    C, S, D, I = 0, 0, 0, 0

    for sample in result_text:
        reference = sample['ref']
        prediction = sample['pred']

        # 处理文本
        reference = remove_special_characters(reference)
        prediction = remove_special_characters(prediction)
        
        # 对于 CER，将句子转换为字符级别
        if cer_wer == 'cer':
            reference = " ".join(list(reference))
            prediction = " ".join(list(prediction))
        
        # 判断字符串是否为空或者空格
        if not reference.strip() or not prediction.strip():
            continue
        
        measures = jiwer.compute_measures(reference, prediction)
        
        C += measures["hits"]
        S += measures["substitutions"]
        D += measures["deletions"]
        I += measures["insertions"]
        total += measures["hits"] + measures["substitutions"] + measures["deletions"]

    cer_wer_score = (S + D + I) / total if total > 0 else 0  # 避免除零错误

    return {f"{cer_wer.upper()}": cer_wer_score, "N": total, "C": C, "S": S, "D": D, "I": I}
