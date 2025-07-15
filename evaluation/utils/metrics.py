#!/usr/bin/env python3
# By xiuwenz2@illinois.edu, Oct. 08, 2024.

import numpy as np
# import dimsim
from pypinyin import lazy_pinyin, Style
import jellyfish, editdistance, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from bert_score import score as bert_score

def calculate_word_error_rate(references, hypotheses):
    distances = 0
    lengths = 0
    for ref, hyp in zip(references, hypotheses):
        r = ref.strip().split()
        h  = hyp.strip().split()
        distance = editdistance.eval(h, r), len(r)
        length = len(r)
        wer = d/l
        distances += distance
        lengths += length
    assert lengths != 0

    return distances/lengths

class SemScore:
    def __init__(self,
                 lang='en',
                 model='R',
                 batch_size=32,
                 device=None,
                 direction='avg',
                 cross_lingual=False,
                 nli_weight=0.4012,
                 bert_weight=0.2785,
                 phonetic_weight=0.3201,
                 **metric_conf):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.lang = lang
        self.batch_size = batch_size
        self.cross_lingual = cross_lingual 
        self.model_type = model
        self.direction = direction 
        self.nli_weight = float(nli_weight)
        self.bert_weight = float(bert_weight)
        self.phonetic_weight = float(phonetic_weight)
        self.metric_config = metric_conf
        self.metric, self.metric_hash = None, None  # Initialize metric (not used here)

        # Initialize NLI model
        self._model, self._tokenizer = self.get_model()

    def collate_input_features(self, pre, hyp):
        tokenized_input_seq_pair = self._tokenizer.encode_plus(pre, hyp,
                                                               max_length=512,
                                                               return_token_type_ids=True, truncation=True)
        input_ids = torch.Tensor(tokenized_input_seq_pair['input_ids']).long().unsqueeze(0).to(self.device)
        token_type_ids = torch.Tensor(tokenized_input_seq_pair['token_type_ids']).long().unsqueeze(0).to(self.device)
        attention_mask = torch.Tensor(tokenized_input_seq_pair['attention_mask']).long().unsqueeze(0).to(self.device)

        return input_ids, token_type_ids, attention_mask

    def score_nli(self, refs, hyps, direction=None, formula='e'):
        direction = direction if direction is not None else self.direction
        # print(f'Computing NLI scores (direction: {direction}, formula: {formula})...')
        
        probs_rh, probs_hr, probs_avg = {}, {}, {}

        with torch.no_grad():
            if direction in ['rh', 'avg']:
                probs = []
                for ref, hyp in zip(refs, hyps):
                    input_ids, token_type_ids, attention_mask = self.collate_input_features(ref, hyp)
                    logits = self._model(input_ids,
                                         attention_mask=attention_mask,
                                         token_type_ids=token_type_ids,
                                         labels=None)[0]
                    prob = torch.softmax(logits, 1).detach().cpu().numpy()
                    probs.append(prob)
                concatenated = np.concatenate(probs, 0)
                # probs_rh['e'], probs_rh['n'], probs_rh['c'] = concatenated[:, 0], concatenated[:, 1], concatenated[:, 2]
                probs_rh['e'], probs_rh['n'] = concatenated[:, 0], concatenated[:, 1]

            if direction in ['hr', 'avg']:
                probs = []
                for ref, hyp in zip(refs, hyps):
                    input_ids, token_type_ids, attention_mask = self.collate_input_features(hyp, ref)
                    logits = self._model(input_ids,
                                         attention_mask=attention_mask,
                                         token_type_ids=token_type_ids,
                                         labels=None)[0]
                    prob = torch.softmax(logits, 1).detach().cpu().numpy()
                    probs.append(prob)
                concatenated = np.concatenate(probs, 0)
                # probs_hr['e'], probs_hr['n'], probs_hr['c'] = concatenated[:, 0], concatenated[:, 1], concatenated[:, 2]
                probs_hr['e'], probs_hr['n'] = concatenated[:, 0], concatenated[:, 1]

            if direction == 'rh':
                final_score = probs_rh['e']
            elif direction == 'hr':
                final_score = probs_hr['e']
            elif direction == 'avg':
                final_score = [(s1+s2)/2.0 for s1, s2 in zip(probs_rh['e'], probs_hr['e'])]
        
        final_score = list(final_score)
        
        return final_score

    def score_all(self, refs, hyps, srcl='en'):
        bert_scores = calculate_bert_score(refs, hyps, self.lang, self.device).tolist() # Bert Score
        # bert_scores = self.min_max_normalize(bert_scores, [0, 1])

        phonetic_scores = []
        for ref, hyp in zip(refs, hyps):
            phonetic_score = calculate_phonetic_similarity(ref, hyp, lang=self.lang)
            phonetic_scores.append(phonetic_score)
        # phonetic_scores = self.min_max_normalize(phonetic_scores, [0.5, 1]) # phonetic score

        nli_scores = self.score_nli(refs, hyps, formula='e')
        nli_scores = self.min_max_normalize(nli_scores, [0, 1])
        
        # combined_scores = [
        #     self.nli_weight * nli + self.bert_weight * bert + self.phonetic_weight * phon
        #     for nli, bert, phon in zip(nli_scores, bert_scores, phonetic_scores)
        # ]
        # 计算平均值
        nli_mean = np.mean(nli_scores)
        bert_mean = np.mean(bert_scores)
        phonetic_mean = np.mean(phonetic_scores)

        combined_scores = {"nli_scores":nli_mean, "bert_scores":bert_mean, "phonetic_scores":phonetic_mean}

        return combined_scores

    def get_model(self):
        if self.lang == 'zh':
            model_path = "/mnt/shareEEx/liuxiaokang/.cache/huggingface/hub/chinese-roberta-wwm-ext-large"
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForSequenceClassification.from_pretrained(model_path)
        elif self.lang == 'en':
            if self.model_type == 'R':
                model_path = "/mnt/shareEEx/liuxiaokang/.cache/huggingface/hub/models--ynie--roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli"
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                model = AutoModelForSequenceClassification.from_pretrained(model_path)
        model.eval()
        model = model.to(self.device)
        return model, tokenizer
    

    def min_max_normalize(self, scores, thre):
        
        assert len(scores) != 0
        normalized_scores = [max(min((score - thre[0]) / (thre[1] - thre[0]), 1), 0) for score in scores]

        return normalized_scores
    
def calculate_bert_score(reference, hypothesis, lang, device):
    if lang == 'zh':
        model_type = "bert-base-chinese"
    elif lang == 'en':
        model_type = "roberta-large"
    P, R, F1 = bert_score(reference, hypothesis, model_type=model_type, lang=lang,rescale_with_baseline=True, device=device)
    return F1


def calculate_phonetic_similarity(reference, hypothesis, lang="en"):
    if lang == 'zh':
        ref_soundex = lazy_pinyin(reference, style=Style.TONE3, neutral_tone_with_five=True)
        hypo_soundex = lazy_pinyin(hypothesis, style=Style.TONE3, neutral_tone_with_five=True)
        ref_soundex = ''.join(ref_soundex)
        hypo_soundex = ''.join(hypo_soundex)
        return jellyfish.jaro_winkler_similarity(hypo_soundex, ref_soundex)
    if lang == 'en':
        hypo_soundex = jellyfish.soundex(hypothesis)
        ref_soundex = jellyfish.soundex(reference)
        return jellyfish.jaro_winkler_similarity(hypo_soundex, ref_soundex)
