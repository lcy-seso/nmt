#!/bin/bash

data_dir=$PWD"/data/"
echo $data_dir

model_dir=$PWD"/models"

if [ -d $model_dir ]; then
  rm -rf $model_dir
else
  mkdir $model_dir
fi
echo $model_dir

shuffle_data=false
CUDA_VISIBLE_DEVICES=0,1,2 python -m nmt.nmt \
  --src=vi --tgt=en \
  --vocab_prefix=$data_dir"/vocab" \
  --train_prefix=$data_dir"/train" \
  --dev_prefix=$data_dir"/tst2013" \
  --test_prefix=$data_dir"/tst2012" \
  --out_dir=$model_dir \
  --triletter_vocab_file="" \
  --eval_label_file="" \
  --jobid=0 \
  --num_train_steps=10000 \
  --steps_per_stats=1 \
  --steps_per_eval=10000 \
  --optimizer="adam" \
  --learning_rate=0.001 \
  --num_gpus=3 \
  --num_layers=4 \
  --num_units=512 \
  --metrics=bleu \
  --residual=True \
  --reg_lambda=1.0 \
  --batch_size=600 \
  --num_buckets=5 \
  --log_device_placement=false \
  --disable_data_shuffle=true \
  --data_parallelism=3 \
  --use_timeline_profiler=true \
  --train_task_only=true \
 # 2>&1 | tee train.log

