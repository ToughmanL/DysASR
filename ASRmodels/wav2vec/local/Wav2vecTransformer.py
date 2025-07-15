# -*- encoding: utf-8 -*-
'''
File       :Wav2vecTransformer.py
Description:
Date       :2025/02/28
Author     :lxk
Version    :1.0
Contact		:xk.liu@siat.ac.cn
License		:GPL
'''

import torch
import torchaudio
from transformers import Wav2Vec2Model, Wav2Vec2Processor
from torch.nn import TransformerDecoder, TransformerDecoderLayer


class Wav2Vec2WithTransformerDecoder(torch.nn.Module):
    def __init__(self, wav2vec2_model_name, decoder_layers, decoder_heads, decoder_dim, vocab_size):
        super(Wav2Vec2WithTransformerDecoder, self).__init__()
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(wav2vec2_model_name)
        decoder_layer = TransformerDecoderLayer(d_model=decoder_dim, nhead=decoder_heads)
        self.decoder = TransformerDecoder(decoder_layer, num_layers=decoder_layers)
        self.fc_out = torch.nn.Linear(decoder_dim, vocab_size)  # vocab_size 是你的词汇表大小

    def forward(self, input_values, labels):
        # Wav2Vec2 encoder
        encoder_outputs = self.wav2vec2(input_values).last_hidden_state

        # Transformer decoder
        decoder_outputs = self.decoder(labels, encoder_outputs)

        # 输出层
        logits = self.fc_out(decoder_outputs)
        return logits
    
    def decode(self, input_values):
        batch_size = encoder_outputs.size(0)
        start_token = torch.zeros(batch_size, 1, dtype=torch.long).to(encoder_outputs.device)  # 假设起始 token 为 0
        decoder_outputs = start_token
        for _ in range(100):  # 最大生成长度
            decoder_outputs = self.decoder(decoder_outputs, encoder_outputs)
            next_token = torch.argmax(self.fc_out(decoder_outputs[:, -1, :]), dim=-1).unsqueeze(1)
            decoder_outputs = torch.cat([decoder_outputs, next_token], dim=1)
            if next_token.item() == 1:  # 假设结束 token 为 1
                break
        logits = self.fc_out(decoder_outputs)
        return logits


def infer(model, audio_file, processor):
    audio_input, sr = torchaudio.load(audio_file)
    input_values = processor(audio_input.squeeze().numpy(), sampling_rate=sr, return_tensors="pt").input_values
    with torch.no_grad():
        logits = model(input_values)
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.decode(predicted_ids[0])
    return transcription


if __name__ == '__main__':
    model_path = "/mnt/shareEEx/liuxiaokang/data/pretrainmodel/wav2vec2-base/wav2vec2-base"
    # 初始化模型
    processor = Wav2Vec2Processor.from_pretrained(model_path)
    vocab_size = processor.tokenizer.vocab_size
    model = Wav2Vec2WithTransformerDecoder(model_path, decoder_layers=6, decoder_heads=8, decoder_dim=165, vocab_size=vocab_size)

    wav_path = "/mnt/shareEEx/liuxiaokang/data/DysarthriaData/CDSD/Data/Aftercutting/Audio/09/S009F001P000W00000.wav"
    audio_input, sr = torchaudio.load(wav_path)

    input_values = processor(audio_input.squeeze().numpy(), sampling_rate=sr, return_tensors="pt").input_values
    labels = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164]).unsqueeze(0)

    logits = model(input_values, labels)
    print(logits.shape)

    transcription = processor.decode(torch.argmax(logits, dim=-1)[0])
    print(transcription)
