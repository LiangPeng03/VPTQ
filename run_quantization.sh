#!/bin/bash
CUDA_VISIBLE_DEVICES=0,1 /home/pengliang/glibc/glibc-2.32-install/lib/ld-linux-x86-64.so.2 \
  --library-path "/home/pengliang/glibc/glibc-2.32-install/lib:/usr/lib64:/usr/lib" \
  $(which python) run_vptq.py \
    --model_name /home/pengliang/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659 \
    --output_dir /home/pengliang/Desktop/test_output/Meta-Llama-3.1-8B-Instruct/ \
    --vector_lens -1 8 \
    --group_num 1 \
    --num_centroids -1 65536 \
    --num_res_centroids -1 256 \
    --npercent 0 \
    --blocksize 128 \
    --eval \
    --new_eval \
    --seq_len 8192 \
    --kmeans_mode hessian \
    --num_gpus 2 \
    --enable_perm \
    --enable_norm \
    --save_model \
    --save_packed_model \
    --hessian_path /home/pengliang/.cache/huggingface/hub/models--VPTQ-community--Hessians-Llama-31-8B-Instruct-6144-8k/snapshots/670ea03c395f95a4f4993603e4aed42a89c974f9 \
    --inv_hessian_path /home/pengliang/.cache/huggingface/hub/models--VPTQ-community--InvHessians-Llama-31-8B-Instruct-6144-8k/snapshots/d56c0d904f25bea48607014ad969e6367279fc27 \
    --ktol 1e-5 --kiter 100

