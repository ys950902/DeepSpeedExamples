#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

# Note that usually LoRA needs to use larger learning rate
OUTPUT_PATH=./output
mkdir -p $OUTPUT_PATH

if [ "$ZERO_STAGE" == "" ]; then
    ZERO_STAGE=2
fi

# Hostfile path
hostfile_deepspeed=/home/yisheng/DeepSpeed-Chat/DeepSpeedExamples/applications/DeepSpeed-Chat/training/hostfile/hostfile_deepspeed
hostfile_mpich=/home/yisheng/DeepSpeed-Chat/DeepSpeedExamples/applications/DeepSpeed-Chat/training/hostfile/hostfile_mpich

# launcher setting
LAUNCHER=${LAUNCHER:-MPICH}
if [[ $LAUNCHER == "deepspeed" ]]; then
        launcher=""
else
        launcher="--force_multi --hostfile $hostfile_deepspeed --launcher=${LAUNCHER} --launcher_args='-hostfile ${hostfile_mpich}'"
fi

CCL=${CCL:-ccl}
run_cmd="
deepspeed $launcher main.py \
   --data_path "/home/yisheng/DeepSpeed-Chat/DeepSpeedExamples/applications/DeepSpeed-Chat/training/dataset/Dahoas/rm-static" "/home/yisheng/DeepSpeed-Chat/DeepSpeedExamples/applications/DeepSpeed-Chat/training/dataset/Dahoas/full-hh-rlhf" "/home/yisheng/DeepSpeed-Chat/DeepSpeedExamples/applications/DeepSpeed-Chat/training/dataset/Dahoas/synthetic-instruct-gptj-pairwise" "/home/yisheng/DeepSpeed-Chat/DeepSpeedExamples/applications/DeepSpeed-Chat/training/dataset/yitingxie/rlhf-reward-datasets" \
   --data_split 2,4,4 \
   --model_name_or_path facebook/opt-1.3b \
   --per_device_train_batch_size 8 \
   --per_device_eval_batch_size 8 \
   --max_seq_len 512 \
   --learning_rate 1e-3 \
   --weight_decay 0.1 \
   --num_train_epochs 1 \
   --gradient_accumulation_steps 1 \
   --lr_scheduler_type cosine \
   --num_warmup_steps 0 \
   --seed 1234 \
   --zero_stage 0 \
   --lora_dim 128 \
   --lora_module_name decoder.layers. \
   --only_optimize_lora \
   --dtype bf16 \
   --deepspeed \
   --output_dir $OUTPUT_PATH \
   2>&1 | tee ./$OUTPUT_PATH/training.log
"
startTime=`date +%Y%m%d-%H:%M:%S`
startTime_s=`date +%s`
echo ${run_cmd}
eval ${run_cmd}
set +x

endTime=`date +%Y%m%d-%H:%M:%S`
endTime_s=`date +%s`
sumTime=$[ $endTime_s - $startTime_s ]
echo "$startTime ---> $endTime" "Total:$sumTime seconds"
