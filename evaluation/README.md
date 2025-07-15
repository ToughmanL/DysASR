# dysarthria_asr

## 环境
conda activate dysasr

## 目的
  + 标准化
    - 数据库和训练测试集规范化
    - 基础方法的规范化
    - 评估方法规范化便于横向对比

  + 发现问题与瓶颈
    - 数据量的缺乏
    - 构音障碍的特异性

## 实验设计
  + 数据库准备
    - 多种数据库 (MSDM CDSD UASPEECH TORGO VoiceBank-2023 SA)

  + 划分方法
    - 说话人不重叠
    - 说话内容不重叠?

  + SOTA方法
    - e2e方法(conformer, transformer)
    - SSL方法(wav2vec, whisper)
    - 语音大模型方法(qwen-audio)

  + 评估方式
    - 使用WER CER衡量外其他衡量方式, bert()
    - 按照不同维度统计结果，（疾病类型、病情程度、音素）


## Dataset
| dataset        | etiology | Participants | Utterances | duration |
|----------------|----------|--------------|------------|----------|
| MSDM           | Stroke   | 83           | 68304      | 36.43h   |
| CDSD           | CP       | 24           | 29466      | 20.1h    |
| VoiceBank-2023 | ALS      | 111          | 12875      | 29.78h   |
| UASPEECH       | CP       | 19           | 143290     | 100h     |
| TORGO          | CP&ALS   | 6            | 8755       | 7.43h    |
| SA             | others   | 291          | 108340      | 278.35 h  |


| Dataset | Language | Etiology | Severity Annotation | Age Range | Gender Ratio (M:F) | No. of Subjects | No. of Utterances | Total Duration |
|---------|----------|------------------------|---------------------|-----------|--------------------|-----------------|-------------------|----------------|
| MSDM~\cite{liu2023audio} | Chinese | Stroke | Yes | 40-85 | 57:26 | 83 | 68,214 | 15.35 h |
| CDSD~\cite{wang2024cdsd} | Chinese | Cerebral Palsy & Others | No | 6-65 | 13:11 | 24 | 29,466 | 20.11 h |
| VoiceBank-2023~\cite{su2023voicebank} | Chinese | ALS | Yes | N/A | 64:47 | 111 | 12,875 | 34.38 h |
| UASPEECH~\cite{kim2008dysarthric} | English | Cerebral Palsy | No | 18-51 | 11:4 | 19 | 143,290 | 100.00 h |
| TORGO~\cite{rudzicz2012torgo} | English | Cerebral Palsy & ALS | Yes | 16-50 | 9:6 | 15 | 8,755 | 7.43 h |
| SAP~\cite{2024Community} | English | 5 Etiologies | Partial | 18+ | N/A | 291 | 108,340 | 278.35 h |


| dataset        | etiology | severity labeling | age range      | male to female ratio | Participants | Utterances | duration |
|----------------|----------|-------------------|----------------|----------------------|--------------|------------|----------|
| MSDM           | Stroke   | Yes               | \              | \                    | 89           | 68304      | 36.43h   |
| CDSD           | CP&OTHER | No                | children&adult | 26/18                | 24           | 29466      | 20.1h    |
| UASPEECH       | CP       | No                | 18-51          | 11/4                 | 15           | 143290     | 100h     |
| TORGO          | CP&ALS   | yes               | 16-50          | 5/3                  | 8            | 8755       | 7.43h    |
| VoiceBank-2023 | ALS      | Yes               | \              | 64/47                | 111          | 12875      | 29.78h   |
| SA             | others   | No                | \              | \                    | 428          | 72179      | 334h     |



|   dataset       | number of people | train | dev | test |
|-----------------|------------------|-------|-----|------|
|  MSDM           | 89               |  77   | 5   |  7   |
|  CDSD           | 24               |  17   | 3   |  4   |
|  UASPEECH       | 19               |  10   | 4   |  5   |
|  TORGO          | 6                |  3    | 1   |  2   |
|  VoiceBank-2023 | 111              |  83   | 14  |  14  |


## Methods
  + conformer_u++
  + paraformer
  + paraformer
  + wav2vec
  + whisper
  + Qwen2-Audio


## Results
| MSDM          | pretrained      | train     | dev                 |       | test                |       |
|---------------|-----------------|-----------|---------------------|-------|---------------------|-------|
|               |                 |           | attention_rescoring | CTC   | attention_rescoring | CTC   |
| conformer_u++ | No              | No        | 68.10               | 69.26 | 86.28               | 86.30 |
| conformer_u++ | No              | Yes       | 14.45               | 16.76 | 36.37               | 40.47 |
| conformer_u++ | Yes             | Yes       | 10.21               | 11.44 | 22.92               | 26.11 |
| paraformer    | No              | No        | 372.64              | ----- | 387.08              | ----- |
| paraformer    | No              | Yes       | 48.51               | ----- | 21.56               | ----- |
| paraformer    | Yes             | Yes       | 7.74                | ----- | 14.26               | ----- |
| wav2vec       | Yes             | No        | 70.04               | ----- | 76.36               | ----- |
| wav2vec       | Yes             | Yes       | 11.17               | ----- | 26.05               | ----- |
| whisper       | Yes             | No        | 148.64              | ----- | 125.98              | ----- |
| whisper       | Yes             | Yes       | 9.52                | ----- | 17.65               | ----- |
| Qwen2-Audio   | Yes             | No        | 169.91              | ----- | 180.56              | ----- |
| Qwen2-Audio   | Yes             | Yes       | 10.56               | ----- | 19.49               | ----- |
|---------------|-----------------|-----------|---------------------|-------|---------------------|-------|
| CDSD          | pretrained      | train     | dev                 |       | test                |       |
|---------------|-----------------|-----------|---------------------|-------|---------------------|-------|
|               |                 |           | attention_rescoring | CTC   | attention_rescoring | CTC   |
| conformer_u++ | No              | No        | 64.25               | 65.48 | 75.99               | 77.07 |
| conformer_u++ | No              | Yes       | 45.27               | 49.70 | 68.49               | 71.30 |
| conformer_u++ | Yes             | Yes       | 26.83               | 30.00 | 56.60               | 58.72 |
| paraformer    | No              | No        | 342.48              | ----- | 417.79              | ----- |
| paraformer    | No              | Yes       | 85.55               | ----- | 106.3               | ----- |
| paraformer    | Yes             | Yes       | 13.73               | ----- | 37.62               | ----- |
| wav2vec       | Yes             | No        | 70.06               | ----- | 76.46               | ----- |
| wav2vec       | Yes             | Yes       | 46.24               | ----- | 62.19               | ----- |
| whisper       | Yes             | No        | 128.60              | ----- | 201.65              | ----- |
| whisper       | Yes             | Yes       | 50.94               | ----- | 68.75               | ----- |
| Qwen2-Audio   | Yes             | No        | 80.63               | ----- | 91.9                | ----- |
| Qwen2-Audio   | Yes             | Yes       | 30.08               | ----- | 43.4                | ----- |
|---------------|-----------------|-----------|---------------------|-------|---------------------|-------|
| VoiceBank     | pretrained      | train     | dev                 |       | test                |       |
|---------------|-----------------|-----------|---------------------|-------|---------------------|-------|
|               |                 |           | attention_rescoring | CTC   | attention_rescoring | CTC   |
| conformer_u++ | No              | No        | 39.07               | 39.87 | 29.30               | 30.16 |
| conformer_u++ | No              | Yes       | 1.19                | 1.86  | 2.07                | 2.84  |
| conformer_u++ | Yes             | Yes       | 0.73                | 0.96  | 1.41                | 1.76  |
| paraformer    | No              | No        | 361.16              | ----- | 334.33              | ----- |
| paraformer    | No              | Yes       | 98.84               | ----- | 98.84               | ----- |
| paraformer    | Yes             | Yes       | 5.17                | ----- | 4.12                | ----- |
| wav2vec       | Yes             | No        | 38.63               | ----- | 30.53               | ----- |
| wav2vec       | Yes             | Yes       | 11.10               | ----- | 8.21                | ----- |
| whisper       | Yes             | No        | 1098.99             | ----- | 857.59              | ----- |
| whisper       | Yes             | Yes       | 3.46                | ----- | 2.59                | ----- |
| Qwen2-Audio   | Yes             | No        | 34.46               | ----- | 38.37               | ----- |
| Qwen2-Audio   | Yes             | Yes       | 1.78                | ----- | 1.93                | ----- |
|---------------|-----------------|-----------|---------------------|-------|---------------------|-------|
| UASPEECH      | pretrained      | train     | dev                 |       | test                |       |
|---------------|-----------------|-----------|---------------------|-------|---------------------|-------|
|               |                 |           | attention_rescoring | CTC   | attention_rescoring | CTC   |
| conformer_u++ | No              | No        | 125.07              | 124.61| 85.97               | 86.30 |
| conformer_u++ | No              | Yes       | 98.55               | 99.33 | 98.57               | 99.35 |
| conformer_u++ | Yes             | Yes       | 35.52               | 39.71 | 22.29               | 25.13 |
| paraformer    | No              | No        | 124.47              | ----- | 118.17              | ----- |
| paraformer    | No              | Yes       | 76.16               | ----- | 73.64               | ----- |
| paraformer    | Yes             | Yes       | 50.49               | ----- | 43.21               | ----- |
| wav2vec       | Yes             | No        | 106.90              | ----- | 133.37              | ----- |
| wav2vec       | Yes             | Yes       | 67.97               | ----- | 57.16               | ----- |
| whisper       | Yes             | No        | 366.17              | ----- | 167.82              | ----- |
| whisper       | Yes             | Yes       | 45.56               | ----- | 35.97               | ----- |
| Qwen2-Audio   | Yes             | No        | 114.59              | ----- | 125.9               | ----- |
| Qwen2-Audio   | Yes             | Yes       | 39.32               | ----- | 33.17               | ----- |
|---------------|-----------------|-----------|---------------------|-------|---------------------|-------|
| TORGO         | pretrained      | train     | dev                 |       | test                |       |
|---------------|-----------------|-----------|---------------------|-------|---------------------|-------|
|               |                 |           | attention_rescoring | CTC   | attention_rescoring | CTC   |
| conformer_u++ | No              | No        | 43.38               | 45.62 | 82.62               | 84.28 |
| conformer_u++ | No              | Yes       | 97.36               | 99.23 | 97.15               | 98.85 |
| conformer_u++ | Yes             | Yes       | 38.93               | 41.71 | 63.70               | 66.34 |
| paraformer    | No              | No        | 115.77              | ----- | 115.22              | ----- |
| paraformer    | No              | Yes       | 100.39              | ----- | 100.98              | ----- |
| paraformer    | Yes             | Yes       | 99.04               | ----- | 87.69               | ----- |
| wav2vec       | Yes             | No        | 31.74               | ----- | 69.79               | ----- |
| wav2vec       | Yes             | Yes       | -----               | ----- | 121.47              | ----- |
| whisper       | Yes             | No        | 45.93               | ----- | 101.74              | ----- |
| whisper       | Yes             | Yes       | 36.46               | ----- | 68.55               | ----- |
| Qwen2-Audio   | Yes             | No        | 100.07              | ----- | 103.62              | ----- |
| Qwen2-Audio   | Yes             | Yes       | 85.25               | ----- | 94.28               | ----- |

