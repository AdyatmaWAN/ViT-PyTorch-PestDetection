#!/bin/bash

# Mendefinisikan daftar file Python yang akan di-looping
batch_pool=("32" "4" "128" "8" "64" "16")
lr_pool=("0.0001" "0.001" "0.00001")
opt_pool=("AdamW" "Adam" "RMSprop")
# Melakukan looping pada setiap file Python dalam daftar
for batch in ${batch_pool[@]}

    do
    for lr in ${lr_pool[@]}
        do
        for opt in ${opt_pool[@]}
            do
                CUDA_VISIBLE_DEVICES=6 python main_bash.py "$batch" "$lr" "$opt"
            done
        done
    done