#! /bin/bash
#
# 【finetune】
#
# 概要:
#      bert-japanise を利用してfinetuning
#
export PRETRAINED_MODEL_PATH='../model/model.ckpt-1400000'
export FINETUNE_OUTPUT_DIR='../model/livedoor_output'

(cd ../src; python3 ./run_classifier.py \
    --task_name=livedoor \
    --do_train=true \
    --do_eval=true \
    --data_dir=/work/data/livedoor \
    --model_file=../model/wiki-ja.model \
    --vocab_file=../model/wiki-ja.vocab \
    --init_checkpoint=${PRETRAINED_MODEL_PATH} \
    --max_seq_length=512 \
    --train_batch_size=4 \
    --learning_rate=2e-5 \
    --num_train_epochs=10 \
    --output_dir=${FINETUNE_OUTPUT_DIR})
