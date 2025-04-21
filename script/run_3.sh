#!/bin/bash
echo "Running job with CUDA_VISIBLE_DEVICES set to: $CUDA_VISIBLE_DEVICES"
cuda_path="cuda_visible_devices.txt"
if [ -f $cuda_path ]; then
  export CUDA_VISIBLE_DEVICES=$(cat $cuda_path)
  num_gpu=$(cat "$cuda_path" | tr ', ' '\n' | grep -c '[0-9]')
  echo "num gpus = $num_gpu"
else
  echo "cuda_visible_devices.txt file not found."
fi

exec bash

# models : gemma2-2/9/27b, qwen-7/14b

# model_name='gemma2-9b-chat'
# # model_name=qwen-7B-chat
# bz=4
# ds_name=arc

# python main.py \
# --model_name $model_name \
# --bz $bz \
# --ds_name $ds_name \


 