# -*- encoding: utf-8 -*-
'''
File       :evaluate.py
Description:
Date       :2025/05/25
Author     :lxk
Version    :1.0
Contact		:xk.liu@siat.ac.cn
License		:GPL
'''

import argparse, os, re, torch, json
from wer_caltulate import compute_wer
from metrics import calculate_word_error_rate, SemScore
from word_unit_split import get_word_unit_text
from tqdm import tqdm

# ignore warnings
import warnings
warnings.filterwarnings("ignore")


# read text file
def read_text(text_path):
    id_text = {}
    with open(text_path, "r") as f:
        for line in f:
            line = line.strip()
            line = line.replace("\t", " ")  # replace tab with space
            ll = line.split(" ", 1)
            if len(ll) == 1:
                key, text = ll[0], ""
            elif len(ll) == 2:
                key, text = ll[0], ll[1]
            else:
                continue
            id_text[key] = text
    return id_text


# pairing hypothesis and reference
def data_pairing(hypo_dict, ref_dict):
    common_keys = hypo_dict.keys() & ref_dict.keys()
    hypo_list = [hypo_dict[k] for k in common_keys]
    ref_list = [ref_dict[k] for k in common_keys]
    return hypo_list, ref_list


# write the result to json
def write_result(output_path, reuslts):
    # write semscores to json
    with open(output_path, 'w') as f:
        json.dump(reuslts, f, indent=4)
    print(f"Results written to {output_path}")


# wer and semscores
def evaluate(hypo_dict, ref_dict, lang):
    hypo_list, ref_list = data_pairing(hypo_dict, ref_dict)
    if len(hypo_list) == 0 or len(ref_list) == 0:
        print("Warning: No common keys found between hypothesis and reference.")
        return {}
    ### Calculating WER
    wer_result = compute_wer(hypo_dict, ref_dict, lang)
    ### Calculating SemScore
    semscores = SemScore(lang=lang).score_all(refs=ref_list, hyps=hypo_list)
    semscores.update(wer_result)
    return semscores


# get level text
def get_level_text(person_list, text_dict):
    person_length = len(person_list[0].split("_"))
    level_text_dict = {}
    for ID, text in text_dict.items():
        person_name = "_".join(ID.split("_")[0:person_length])
        if person_name in person_list:
            level_text_dict[ID] = text
    return level_text_dict


# severity level WER
def severity_level_wer(hypo_dict, ref_dict, level_file, part, lang):
    level_result = {}
    with open(level_file, 'r') as f:
        level_dict = json.load(f)
    for level_name, level_info in level_dict.items():
        person_list = level_info.get(part, [])
        if len(person_list) == 0:
            continue
        level_ref_dict = get_level_text(person_list, ref_dict)
        level_hypo_dict = get_level_text(person_list, hypo_dict)
        wer_result = compute_wer(level_hypo_dict, level_ref_dict, lang)
        level_result[level_name] = wer_result
        # print(f"Severity Level: {level_name}, WER: {wer_result}")
    return level_result


# word unit WER
def word_unit_wer(hypo_dict, ref_dict, dataset, lang):
    word_units_result = {}
    word_unit_text_dict = get_word_unit_text(ref_dict, dataset)
    for unit_name, unit_ref_dict in word_unit_text_dict.items():
        unit_hypo_dict = {k: hypo_dict[k] for k in unit_ref_dict.keys() if k in hypo_dict}
        wer_result = compute_wer(unit_hypo_dict, unit_ref_dict,lang)
        word_units_result[unit_name] = wer_result
        print(f"Word Unit: {unit_name}, length:{len(unit_ref_dict)} WER: {wer_result}")
    return word_units_result


# main function
def interface(config_file, severity_level=False, word_unit=False):
    # read json of list
    with open(config_file, 'r') as f:
        config = json.load(f)
    results = []
    for dataset, conf_info in config.items():
        lang = conf_info.get("lang")
        level_file = conf_info.get("level_file")
        for part in ["test", "dev"]:
            part_info = conf_info.get(part, {})
            for model, path_info in part_info.items():
                root_path = path_info.get("root")
                ref_path = os.path.join(root_path, path_info.get("ref_path"))
                for hypo_path in path_info.get("decode", []):
                    hypo_path = os.path.join(root_path, hypo_path)
                    result = {"dataset": dataset, "lang": lang, "part": part, "model": model, "hypo_path": hypo_path}
                    if not os.path.exists(hypo_path):
                        print(f"Error: Hypothesis file {hypo_path} does not exist.")
                        continue
                    if model == "whisper":
                        ref_path = hypo_path.replace("decode", "label")
                    ref_dict = read_text(ref_path)
                    hypo_dict = read_text(hypo_path)
                    if severity_level:
                        if  level_file is None:
                            continue
                        semscores = severity_level_wer(hypo_dict, ref_dict, level_file, part, lang)
                    elif word_unit:
                        if dataset == "uaspeech":
                            continue
                        semscores = word_unit_wer(hypo_dict, ref_dict, dataset, lang)
                    else:
                        semscores = evaluate(hypo_dict, ref_dict, lang)
                    # 判断semscores是否为{}
                    if not semscores:
                        print(f"Warning: No scores calculated for {hypo_path}.")
                        continue
                    result.update(semscores)
                    print(result)
                    results.append(result)
    print("Evaluation completed.")
    return results


# entry point
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the hypothesis and reference")
    parser.add_argument("--result_path", type=str, default="results/eval.json", help="Path to save the evaluation results")
    parser.add_argument("--config_file", type=str, required=True, help="Path to the hypothesis file")
    # compute severity level wer
    parser.add_argument("--severity_level", action='store_true', help="Compute severity level WER")
    # compute word unit wer
    parser.add_argument("--word_unit_wer", action='store_true', help="Compute syllable\word\sentence unit WER")
    args = parser.parse_args()
    config_file = args.config_file
    results_path = args.result_path
    results = interface(config_file, args.severity_level, args.word_unit_wer)
    write_result(results_path, results)