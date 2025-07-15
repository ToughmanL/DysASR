# CDSD: Chinese Dysarthria Speech Database

## Abstract：
We present the Chinese Dysarthria Speech Database (CDSD) as a valuable resource for dysarthria research. This database comprises speech data from 24 participants with dysarthria. Among these participants, one recorded an additional 10 hours of speech data, while each recorded one hour, resulting in 34 hours of speech material. To accommodate participants with varying cognitive levels, our text pool primarily consists of content from the AISHELL-1 dataset and speeches by primary and secondary school students. When participants read these texts, they must use a mobile device or the ZOOM F8n multi-track field recorder to record their speeches. In this paper, we elucidate the data collection and annotation processes and present an approach for establishing a baseline for dysarthric speech recognition. Furthermore, we conducted a speaker-dependent dysarthric speech recognition experiment using an additional 10 hours of speech data from one of our participants. Our research findings indicate that, through extensive data-driven model training, fine-tuning limited quantities of specific individual data yields commendable results in speaker-dependent dysarthric speech recognition. However, we observe significant variations in recognition results among different dysarthric speakers. These insights provide valuable reference points for speaker-dependent dysarthric speech recognition.


Raw data consists of unprocessed speech and video data. The numbering corresponds to the participant ID, not sequential numbering. Participants with IDs 01, 02, 04, 07, 09, 10, 13, 15, 16, 18, 23 are female, while the rest are male. Participants with IDs 1 6, 17, 18, 19, and 20 recorded using professional recording equipment, while others used mobile devices. Participants with IDs 01, 06, and 18 are underage (below 18 years old), while the rest are adults. Participant 20's folder includes 1 hour of speech da ta and 10 hours of speech data used for testing. Each folder for a specific ID contains audio or video data for that participant, along with reference text. However, participants may have yet to read the text in the order provided. After cutting, we align the audio with the textual content, enabling researchers to understand clearly what material the participants are reading.

## 
实际上可以使用语音数据有24个人包括（01, 02, 03, 04, 05, 06, 07, 08, 09, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 23, 25, 26）
男性：(01, 02, 04, 07, 09, 10, 13, 15, 16, 18, 23)
女性：(03, 05, 06, 08, 11, 12, 14, 17, 19, 20, 21, 25, 26)
专业设备录音: (16, 17, 18, 19, 20)
移动设备录音: (01, 02, 03, 04, 05, 06, 07, 08, 09, 10, 11, 12, 13, 14, 15, 21, 23, 25, 26)
未成年人: (01, 06, 18)
成年人:(02, 03, 04, 05, 07, 08, 09, 10, 11, 12, 13, 14, 15, 16, 17, 19, 20, 21, 23, 25, 26)
在本实验中将(01 02 03 05 17)作为测试集，其余作为训练集


## Citation
Sun, Mengyi, et al. "CDSD: Chinese Dysarthria Speech Database." arXiv preprint arXiv:2310.15930 (2023).