#!/bin/bash
stage=3

# datasets=("CDSD" "MSDM" "VoiceBank" "TORGO" "UASPEECH" "SpeechAccessibility")
datasets=("SpeechAccessibility")
zh_datasets=("CDSD" "MSDM" "VoiceBank")
en_datasets=("TORGO" "UASPEECH" "SpeechAccessibility")
gpus=("0" "1") # "0" "1" "2" "3"

# dataset preparation
if [ ${stage} -eq 0 ]; then
    python tools/datalist_convert.py  ~/workspace/wenet/wenet-3.1.0/examples/msdm/250225/data dataset/MSDM Chinese
    python tools/datalist_convert.py  ~/workspace/wenet/wenet-3.1.0/examples/cdsd/250224/data/ dataset/CDSD Chinese
    python tools/datalist_convert.py  ~/workspace/wenet/wenet-3.1.0/examples/voicebank/250224/data/ dataset/VoiceBank Chinese
    python tools/datalist_convert.py  ~/workspace/wenet/wenet-3.1.0/examples/TROGO/250224/data/ dataset/TORGO English
    python tools/datalist_convert.py  ~/workspace/wenet/wenet-3.1.0/examples/uaspeech/250225/data/ dataset/UASPEECH English
    python tools/datalist_convert.py  ~/workspace/wenet/wenet-3.1.0/examples/SpeechAccessibility/conformer/data/ dataset/SpeechAccessibility English
fi

# finetune
if [ ${stage} -eq 1 ]; then
    echo "finetune"
    for index in ${!datasets[@]}; do
        dataset=${datasets[$index]}
        # gpu=${gpus[$index]}
        gpu=2
        echo "CUDA_VISIBLE_DEVICES=${gpu} nohup python finetune.py --train_data dataset/${dataset}/train.json --test_data dataset/${dataset}/dev.json --output_dir output/${dataset}/" >> log/${dataset}_fintune.log;
        CUDA_VISIBLE_DEVICES=${gpu} nohup python finetune.py --train_data dataset/${dataset}/train.json --test_data dataset/${dataset}/dev.json --output_dir output/${dataset}/ 1>>log/${dataset}_fintune.log 2>&1 &
    done
fi

# merge lora
if [ ${stage} -eq 2 ]; then
    echo "merge lora"
    for index in ${!datasets[@]}; do
        dataset=${datasets[$index]}
        gpu=${gpus[$index]}
        CUDA_VISIBLE_DEVICES=${gpu} nohup python merge_lora.py --lora_model=output/${dataset}/whisper-base/checkpoint-best --output_dir=models/${dataset} 1>>log/${dataset}_fintune.log 2>&1 &
    done
fi

# evaluation
if [ ${stage} -eq 3 ]; then
    echo "evaluation"

    for index in ${!datasets[@]}; do
        dataset=${datasets[$index]}
        exists=false
        for zh_dataset in ${zh_datasets[@]}; do
            if [ ${dataset} == ${zh_dataset} ]; then
                exists=true
                break
            fi
        done
        if [ ${exists} == true ]; then
            language=Chinese
            metric=cer
        else
            language=English
            metric=wer
        fi

        decode_methods=("finetune" "nofinetune")
        for decode_method in ${decode_methods[@]}; do
            gpu=${gpus[$index]}
            if [ ${decode_method} == "finetune" ]; then
                echo "decode_method: ${decode_method}"
                model_path=models/${dataset}/whisper-base-finetune
                result_dir=results/${dataset}
            else
                echo "decode_method: ${decode_method}"
                model_path=/mnt/shareEEx/liuxiaokang/data/pretrainmodel/whisper-base
                result_dir=init_results/${dataset}
            fi
            echo "CUDA_VISIBLE_DEVICES=${gpu} python evaluation.py --test_data dataset/${dataset}/dev.json dataset/${dataset}/test.json --model_path=$model_path --result_dir $result_dir --language $language --metric $metric"
            CUDA_VISIBLE_DEVICES=${gpu} nohup python evaluation.py --test_data dataset/${dataset}/dev.json dataset/${dataset}/test.json --model_path=$model_path --result_dir $result_dir --language $language --metric $metric 1>>log/${dataset}_${decode_method}.log 2>&1 &
        done
    done
fi




# CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 finetune.py --base_model=openai/whisper-tiny --use_8bit=False --per_device_train_batch_size=8 --per_device_eval_batch_size=8 --gradient_accumulation_steps=1
# CUDA_VISIBLE_DEVICES=0 python merge_lora.py --lora_model=output/whisper-tiny/checkpoint-final

# CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 finetune.py --base_model=openai/whisper-base --use_8bit=False --per_device_train_batch_size=8 --per_device_eval_batch_size=8 --gradient_accumulation_steps=1
# CUDA_VISIBLE_DEVICES=0 python merge_lora.py --lora_model=output/whisper-base/checkpoint-final

# CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 finetune.py --base_model=openai/whisper-small --use_8bit=True --per_device_train_batch_size=8 --per_device_eval_batch_size=8 --gradient_accumulation_steps=1
# CUDA_VISIBLE_DEVICES=0 python merge_lora.py --lora_model=output/whisper-small/checkpoint-final

# CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 finetune.py --base_model=openai/whisper-medium --use_8bit=True --per_device_train_batch_size=4 --per_device_eval_batch_size=2 --gradient_accumulation_steps=2
# CUDA_VISIBLE_DEVICES=0 python merge_lora.py --lora_model=output/whisper-medium/checkpoint-final

# CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 finetune.py --base_model=openai/whisper-large-v2 --use_8bit=True --per_device_train_batch_size=2 --per_device_eval_batch_size=2 --gradient_accumulation_steps=4
# CUDA_VISIBLE_DEVICES=0 python merge_lora.py --lora_model=output/whisper-large-v2/checkpoint-final


# CUDA_VISIBLE_DEVICES=0 python evaluation.py --model_path=models/whisper-tiny-finetune
# CUDA_VISIBLE_DEVICES=0 python evaluation.py --model_path=models/whisper-base-finetune
# CUDA_VISIBLE_DEVICES=0 python evaluation.py --model_path=models/whisper-small-finetune
# CUDA_VISIBLE_DEVICES=0 python evaluation.py --model_path=models/whisper-medium-finetune
# CUDA_VISIBLE_DEVICES=0 python evaluation.py --model_path=models/whisper-large-v2-finetune


# CUDA_VISIBLE_DEVICES=0 nohup python finetune.py --train_data dataset/CDSD/train.json --test_data dataset/CDSD/dev.json --output_dir output/CDSD/ 1>log/CDSD_fintune.log 2>&1 &
# CUDA_VISIBLE_DEVICES=0 nohup python merge_lora.py --lora_model=output/CDSD/whisper-base/checkpoint-best --output_dir=models/CDSD 1>>log/CDSD_fintune.log 2>&1 &
# CUDA_VISIBLE_DEVICES=0 nohup python evaluation.py --test_data dataset/CDSD/test.json --model_path=models/CDSD --language Chinese --metric cer 1>>log/CDSD_fintune.log 2>&1 &
# CUDA_VISIBLE_DEVICES=0 python evaluation.py --test_data dataset/TROGO/test.json dataset/TROGO/dev.json --model_path=models/TORGO/whisper-base-finetune/ --result_dir results/TROGO --language English --metric wer