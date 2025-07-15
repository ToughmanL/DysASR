# -*- encoding: utf-8 -*-
'''
File       :log2csv.py
Description:
Date       :2025/06/03
Author     :lxk
Version    :1.0
Contact		:xk.liu@siat.ac.cn
License		:GPL
'''

import csv
import json


# read text
def read_text_file(file_path):
    lines = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if len(line) > 0:
                lines.append(line)
    return lines


def convert_text_to_csv(lines, output_file):
    # 解析每个字典
    data = []
    for line in lines:
        try:
            # 移除单引号，使字符串符合JSON格式
            json_line = line.replace("'", "\"")
            # 解析JSON字符串
            data.append(json.loads(json_line))
        except json.JSONDecodeError as e:
            print(f"Error decoding line: {line}")
            print(f"Error: {e}")
    
    # 获取所有可能的字段名
    fieldnames = set()
    for item in data:
        fieldnames.update(item.keys())
    fieldnames = sorted(fieldnames)
    
    # 写入CSV文件
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        # 写入表头
        writer.writeheader()
        # 写入数据
        writer.writerows(data)
    
    print(f"CSV file has been created at: {output_file}")

# 示例用法
if __name__ == "__main__":
    input_text = "log/eval_SAP_regu.log"
    output_csv = "results/eval_SAP_regu.csv"
    lines = read_text_file(input_text)
    convert_text_to_csv(lines, output_csv)
    print("Conversion completed successfully.")