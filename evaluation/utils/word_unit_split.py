# -*- encoding: utf-8 -*-
'''
File       :word_unit_split.py
Description:
Date       :2025/06/02
Author     :lxk
Version    :1.0
Contact		:xk.liu@siat.ac.cn
License		:GPL
'''

import re

# MSDM, CDSD and VoiceBank word unit text
def get_chinese_word_unit_text(text_dict):
    word_unit_text_dict = {"character":{}, "word":{}, "phrase":{}, "sentence":{}}
    for ID, text in text_dict.items():
      pattern = re.compile(r'[^\u4e00-\u9fa5a-zA-Z0-9]')
      text = pattern.sub(" ", text).strip()
      text_len = len(text)
      if text_len == 0:
        word_unit_text_dict['character'][ID] = " "
      elif text_len == 1:
        word_unit_text_dict['character'][ID] = text
      elif text_len == 2:
        word_unit_text_dict['word'][ID] = text
      elif text_len >2 and text_len <= 5:
        word_unit_text_dict['phrase'][ID] = text
      elif text_len > 5:
        word_unit_text_dict['sentence'][ID] = text
    return word_unit_text_dict

# TORGO and UASpeech word unit text
def get_english_word_unit_text(text_dict):
    word_unit_text_dict = {"word":{}, "sentence":{}}
    for ID, text in text_dict.items():
        pattern = re.compile(r'[^a-zA-Z0-9]')
        text = pattern.sub(" ", text).strip()
        text_len = len(text.split())
        if text_len == 0:
            word_unit_text_dict['word'][ID] = " "
        elif text_len == 1:
            word_unit_text_dict['word'][ID] = text
        elif text_len > 1:
            word_unit_text_dict['sentence'][ID] = text
    return word_unit_text_dict


# get word unit text
def get_word_unit_text(text_dict, dataset):
    if dataset in ["msdm", "cdsd", "voicebank"]:
        return get_chinese_word_unit_text(text_dict)
    elif dataset in ["torgo", "uaspeech"]:
        return get_english_word_unit_text(text_dict)
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")