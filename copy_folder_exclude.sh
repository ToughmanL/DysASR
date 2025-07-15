#!/bin/bash

#!/bin/bash


stage=4

if [ $stage -eq 1 ]; then
    # 定义源和目标路径的数组
    SRC_DIRS=(
        "/mnt/shareEEx/liuxiaokang/workspace/selfsupervised/fairseq/examples/wav2vec"
        "/mnt/shareEEx/liuxiaokang/workspace/wenet/wenet-3.1.0"
        "/mnt/shareEEx/liuxiaokang/workspace/funasr/FunASR-250301"
        "/mnt/shareEEx/liuxiaokang/workspace/whisper/Whisper-Finetune-250225"
        "/mnt/shareEEx/liuxiaokang/workspace/LLM/Qwen2-Audio"
    )

    DST_DIRS=(
        "ASRmodels/wav2vec"
        "ASRmodels/wenet"
        "ASRmodels/FunASR"
        "ASRmodels/Whisper-Finetune"
        "ASRmodels/Qwen2-Audio"
    )

    # 遍历并同步每对路径
    for i in "${!SRC_DIRS[@]}"; do
        SRC="${SRC_DIRS[$i]}"
        DST="${DST_DIRS[$i]}"

        echo "🔄 正在同步: $SRC -> $DST"
        mkdir -p "$DST"

        rsync -avP --partial \
            --exclude='*.log' \
            --exclude='*.pt' \
            --exclude='*.pth' \
            --exclude='*.wav' \
            --exclude='*.txt' \
            --exclude='*.ckpt' \
            --exclude='*.onnx' \
            --exclude='*.bin' \
            --exclude='*.safetensors' \
            --exclude='onnxruntime/' \
            --exclude='examples/' \
            --exclude='model_zoo/' \
            --exclude='.git/' \
            "$SRC/" "$DST/"

        echo "✅ 完成同步: $SRC -> $DST"
        echo "---------------------------------------------"
    done

    echo "🎉 所有模型目录同步完成！"

elif [ $stage -eq 2 ]; then
    # 定义源和目标路径的数组
    SRC_DIRS=(
        "/mnt/shareEEx/liuxiaokang/workspace/funasr/FunASR-250301/examples/UASpeech"
        "/mnt/shareEEx/liuxiaokang/workspace/funasr/FunASR-250301/examples/CDSD"
        "/mnt/shareEEx/liuxiaokang/workspace/funasr/FunASR-250301/examples/MSDM"
        "/mnt/shareEEx/liuxiaokang/workspace/funasr/FunASR-250301/examples/SpeechAccessibility"
        "/mnt/shareEEx/liuxiaokang/workspace/funasr/FunASR-250301/examples/VoiceBank"
        "/mnt/shareEEx/liuxiaokang/workspace/funasr/FunASR-250301/examples/TROGO"

    )

    DST_DIRS=(
        "ASRmodels/FunASR/example/UASpeech"
        "ASRmodels/FunASR/example/CDSD"
        "ASRmodels/FunASR/example/MSDM"
        "ASRmodels/FunASR/example/SpeechAccessibility"
        "ASRmodels/FunASR/example/VoiceBank"
        "ASRmodels/FunASR/example/TROGO"
    )

    # 遍历并同步每对路径
    for i in "${!SRC_DIRS[@]}"; do
        SRC="${SRC_DIRS[$i]}"
        DST="${DST_DIRS[$i]}"

        echo "🔄 正在同步: $SRC -> $DST"
        mkdir -p "$DST"

        rsync -avP --partial \
            --exclude='*.log' \
            --exclude='*.pt' \
            --exclude='*.pth' \
            --exclude='*.wav' \
            --exclude='*.txt' \
            --exclude='*.ckpt' \
            --exclude='*.onnx' \
            --exclude='*.bin' \
            --exclude='*.safetensors' \
            --exclude='onnxruntime/' \
            --exclude='examples/' \
            --exclude='model_zoo/' \
            --exclude='.git/' \
            --exclude='pretrained_model/' \
            --exclude='exp/' \
            "$SRC/" "$DST/"

        echo "✅ 完成同步: $SRC -> $DST"
        echo "---------------------------------------------"
    done

    echo "🎉 所有模型目录同步完成！"
elif [ $stage -eq 3 ]; then
    # 定义源和目标路径的数组
    SRC_DIRS=(
        "/mnt/shareEEx/liuxiaokang/workspace/wenet/wenet-3.1.0/examples/cdsd/250224"
        "/mnt/shareEEx/liuxiaokang/workspace/wenet/wenet-3.1.0/examples/msdm/250225"
        "/mnt/shareEEx/liuxiaokang/workspace/wenet/wenet-3.1.0/examples/SpeechAccessibility/250225"
        "/mnt/shareEEx/liuxiaokang/workspace/wenet/wenet-3.1.0/examples/TROGO/250224"
        "/mnt/shareEEx/liuxiaokang/workspace/wenet/wenet-3.1.0/examples/uaspeech/250225"
        "/mnt/shareEEx/liuxiaokang/workspace/wenet/wenet-3.1.0/examples/voicebank/250224"
    )

    DST_DIRS=(
        "ASRmodels/wenet/example/CDSD/conformer"
        "ASRmodels/wenet/example/MSDM/conformer"
        "ASRmodels/wenet/example/SpeechAccessibility/conformer"
        "ASRmodels/wenet/example/TROGO/conformer"
        "ASRmodels/wenet/example/UASpeech/conformer"
        "ASRmodels/wenet/example/VoiceBank/conformer"
    )

    # 遍历并同步每对路径
    for i in "${!SRC_DIRS[@]}"; do
        SRC="${SRC_DIRS[$i]}"
        DST="${DST_DIRS[$i]}"

        echo "🔄 正在同步: $SRC -> $DST"
        mkdir -p "$DST"

        rsync -avP --partial \
            --exclude='*.log' \
            --exclude='*.slice' \
            --exclude='*.shape' \
            --exclude='*.pt' \
            --exclude='*.pth' \
            --exclude='*.wav' \
            --exclude='*.txt' \
            --exclude='*.ckpt' \
            --exclude='*.onnx' \
            --exclude='*.bin' \
            --exclude='*.safetensors' \
            --exclude='onnxruntime/' \
            --exclude='examples/' \
            --exclude='model_zoo/' \
            --exclude='.git/' \
            --exclude='pretrained_model/' \
            --exclude='exp/' \
            --exclude='tensorboard/' \
            --exclude='audio/' \
            --exclude='log/' \
            "$SRC/" "$DST/"

        echo "✅ 完成同步: $SRC -> $DST"
        echo "---------------------------------------------"
    done

    echo "🎉 所有模型目录同步完成！"
elif [ $stage -eq 4 ]; then
    # 定义源和目标路径的数组
    SRC_DIRS=(
        "/mnt/shareEEx/liuxiaokang/workspace/DysarthriaASR"
    )

    DST_DIRS=(
        "evaluation"
    )

    # 遍历并同步每对路径
    for i in "${!SRC_DIRS[@]}"; do
        SRC="${SRC_DIRS[$i]}"
        DST="${DST_DIRS[$i]}"

        echo "🔄 正在同步: $SRC -> $DST"
        mkdir -p "$DST"

        rsync -avP --partial \
            --exclude='*.log' \
            --exclude='*.pt' \
            --exclude='*.pth' \
            --exclude='*.wav' \
            --exclude='*.txt' \
            --exclude='*.ckpt' \
            --exclude='*.onnx' \
            --exclude='*.bin' \
            --exclude='*.safetensors' \
            --exclude='onnxruntime/' \
            --exclude='examples/' \
            --exclude='model_zoo/' \
            --exclude='.git/' \
            --exclude='dataroot/' \
            --exclude='docs/' \
            --exclude='models--bert-base-chinese/' \
            --exclude='models--ynie--roberta-large-snli_mnli_fever_anli_R1_R
            2_R3-nli/' \
            --exclude='plots/' \
            --exclude='results/' \
            --exclude='roberta-large/' \
            --exclude='.cache/' \
            "$SRC/" "$DST/"

        echo "✅ 完成同步: $SRC -> $DST"
        echo "---------------------------------------------"
    done

    echo "🎉 所有模型目录同步完成！"
fi
