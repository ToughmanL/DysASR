#!/bin/bash

stage=2
root_data_dir=/mnt/shareEEx/liuxiaokang/workspace/selfsupervised/fairseq/examples/wav2vec/
data_set=SpeechAccessibility

if [ $stage -eq 1 ]; then
  # Prepare data
  fairseq-preprocess  --source-lang ltr --target-lang ltr --trainpref ${root_data_dir}/data/${data_set}/train --validpref ${root_data_dir}/data/${data_set}/dev --testpref ${root_data_dir}/data/${data_set}/test --destdir ${root_data_dir}/data/${data_set} --workers 4
  # cp ${root_data_dir}/data-bin/${data_set}/dict.ltr.txt ${root_data_dir}/data/${data_set}/dict.ltr.txt
fi


PORT=5000

if [ $stage -eq 2 ]; then
  # Train model
  python wav2vec_fintune_iter.py;
  python fintune_main_iter.py;
fi


if [ $stage -eq 3 ]; then
  # Evaluate model
  python local/wav2vec_infer.py;
fi