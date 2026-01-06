#!/bin/bash
CUDA_VISIBLE_DEVICES=0 /home/pengliang/glibc/glibc-2.32-install/lib/ld-linux-x86-64.so.2 \
  --library-path "/home/pengliang/glibc/glibc-2.32-install/lib:/usr/lib64:/usr/lib" \
  $(which python) run_vptq_opt.py \
    --model_name /home/pengliang/.cache/huggingface/hub/models--facebook--opt-125m/snapshots/27dcfa74d334bc871f3234de431e71c6eeba5dd6 \
    --output_dir /home/pengliang/Desktop/test_output/opt_125m/ \
    --vector_lens -1 8 \
    --group_num 1 \
    --num_centroids -1 65536 \
    --num_res_centroids -1 256 \
    --npercent 0 \
    --blocksize 128 \
    --new_eval \
    --seq_len 2048 \
    --kmeans_mode hessian \
    --num_gpus 2 \
    --enable_perm \
    --enable_norm \
    --save_model \
    --save_packed_model \
    --hessian_path /home/pengliang/Desktop/local_models/opt_125m/hessians \
    --inv_hessian_path /home/pengliang/Desktop/local_models/opt_125m/inv_hessians \
    --ktol 1e-5 --kiter 100