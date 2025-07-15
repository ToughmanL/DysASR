#!/bin/bash

stage=1

# train
if [ $stage -eq 0 ]; then
  echo "Stage 0: Train"
  python train.py --config config/train.yaml
fi

# eval
if [ $stage -eq 1 ]; then
  python utils/evaluate.py --config_file conf/test.json --result_path results/eval_test.json
fi

# eval with severity eqvel
if [ $stage -eq 2 ]; then
  python utils/evaluate.py --config_file conf/test_cdsd.json --result_path results/eval_cdsd_results_severity.json --severity_level 
fi

# eval with word unit
if [ $stage -eq 3 ]; then
  python utils/evaluate.py --config_file conf/test_cdsd.json --result_path results/eval_cdsd_results_units.json --word_unit_wer 
fi
