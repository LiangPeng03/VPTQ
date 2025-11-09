#!/bin/bash
CUDA_VISIBLE_DEVICES=0,1 /home/pengliang/glibc/glibc-2.32-install/lib/ld-linux-x86-64.so.2 \
  --library-path "/home/pengliang/glibc/glibc-2.32-install/lib:/usr/lib64:/usr/lib" \
  $(which python) run_vptq.py \
    --model_name /home/pengliang/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3-8B/snapshots/8cde5ca8380496c9a6cc7ef3a8b46a0372a1d920 \
    --output_dir /home/pengliang/Desktop/test_output/Meta-Llama-3-8b/ \
    --vector_lens -1 8 \
    --group_num 1 \
    --num_centroids -1 65536 \
    --num_res_centroids -1 256 \
    --npercent 0 \
    --blocksize 128 \
    --new_eval \
    --seq_len 8192 \
    --kmeans_mode hessian \
    --num_gpus 2 \
    --enable_perm \
    --enable_norm \
    --save_model \
    --save_packed_model \
    --hessian_path /home/pengliang/.cache/huggingface/hub/models--marcusguhao--llama-3-8b-hessian/snapshots/000bcdb231570df04a4d5f34bb3cc72cb82a8c6c \
    --inv_hessian_path /home/pengliang/Desktop/local_models/llama3_8b/inv_hessians \
    --ktol 1e-5 --kiter 100

