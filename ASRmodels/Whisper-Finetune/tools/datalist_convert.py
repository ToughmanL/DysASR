# -*- encoding: utf-8 -*-
'''
File       :datalist_convert.py
Description:
Date       :2025/02/26
Author     :lxk
Version    :1.0
Contact		:xk.liu@siat.ac.cn
License		:GPL
'''

import os
import json
import soundfile
import argparse

def DatalistConvert(datalist, datajson, language):
    new_lines = []
    with open(datalist, 'r', encoding='utf-8') as rf:
        lines = rf.readlines()
        for line in lines:
            line = line.strip()
            info = json.loads(line)
            new_info = {"audio": {"path": info["wav"]}, "sentence": info["txt"], "speaker": info["speaker"], "language": language, "key": info["key"]}
            if "dur" in info:
                duration = info["dur"]
            elif "duration" in info:
                duration = info["duration"]
            else:
                sample, sr = soundfile.read(info["wav"])
                duration = round(max(sample.shape) / float(sr), 2)
            text = info["txt"].strip()
            # 判断text中是否有有效内容
            if not text or len(text) == 0 or text == " ":
                print(f"Warning: The text for {info['wav']} is empty or whitespace. Skipping this entry.")
                continue
            new_info["duration"] = duration
            new_info["sentences"] = [{"start": 0, "end": duration, "text": text}]
            new_lines.append(new_info)
    with open(datajson, 'w', encoding='utf-8') as wf:
        for line in new_lines:
            wf.write(json.dumps(line, ensure_ascii=False) + "\n")
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert datalist to json format")
    parser.add_argument("wenet_datadir", type=str, help="datalist file")
    parser.add_argument("datadir", type=str, help="new datalist file")
    parser.add_argument("language", type=str, help="language")
    args = parser.parse_args()

    if not os.path.exists(args.wenet_datadir):
        raise FileNotFoundError(f"The directory {args.wenet_datadir} does not exist.")
    train_data_list = os.path.join(args.wenet_datadir, "train", "data.raw.list")
    test_data_list = os.path.join(args.wenet_datadir, "test", "data.raw.list")
    dev_data_list = os.path.join(args.wenet_datadir, "dev", "data.raw.list")

    if not os.path.exists(args.datadir):
        os.makedirs(args.datadir)
    new_train_json = os.path.join(args.datadir, "train.json")
    new_test_json = os.path.join(args.datadir, "test.json")
    new_dev_json = os.path.join(args.datadir, "dev.json")

    language = args.language

    DatalistConvert(dev_data_list, new_dev_json, language)
    DatalistConvert(test_data_list, new_test_json, language)
    DatalistConvert(train_data_list, new_train_json, language)
    
'''
python tools/datalist_convert.py  ~/workspace/wenet/wenet-3.1.0/examples/msdm/250225/data dataset/MSDM Chinese
python tools/datalist_convert.py  ~/workspace/wenet/wenet-3.1.0/examples/cdsd/250224/data/ dataset/CDSD Chinese
python tools/datalist_convert.py  ~/workspace/wenet/wenet-3.1.0/examples/voicebank/250224/data/ dataset/VoiceBank Chinese
python tools/datalist_convert.py  ~/workspace/wenet/wenet-3.1.0/examples/TROGO/250224/data/ dataset/TORGO English
python tools/datalist_convert.py  ~/workspace/wenet/wenet-3.1.0/examples/uaspeech/250225/data/ dataset/UASPEECH English
'''