#!/bin/bash

# Mendefinisikan daftar file Python yang akan di-looping
batch_pool=("32" "2" "128" "8")
lr_pool=("0.0001" "0.001" "0.00001")
opt_pool=("AdamW" "Adam" "RMSprop" "Lion" "SGD")
# Melakukan looping pada setiap file Python dalam daftar
for batch in ${batch_pool[@]}

    do
    for lr in ${lr_pool[@]}
        do
        for opt in ${opt_pool[@]}
            do
                python main_bash.py "$batch" "$lr" "$opt"
            done
        done
    done