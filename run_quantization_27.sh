#!/bin/bash
CUDA_VISIBLE_DEVICES=0,1 /home/pengliang/glibc/glibc-2.32-install/lib/ld-linux-x86-64.so.2 \
  --library-path "/home/pengliang/glibc/glibc-2.32-install/lib:/usr/lib64:/usr/lib" \
  $(which python) run_vptq.py \
    --model_name /home/pengliang/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-hf/snapshots/01c7f73d771dfac7d292323805ebc428287df4f9 \
    --output_dir /home/pengliang/Desktop/test_output/Meta-Llama-2-7b/ \
    --vector_lens -1 6 \
    --group_num 1 \
    --num_centroids -1 4096 \
    --num_res_centroids -1 -1 \
    --npercent 0 \
    --blocksize 128 \
    --new_eval \
    --seq_len 2048 \
    --kmeans_mode hessian \
    --num_gpus 2 \
    --enable_perm True \
    --enable_norm True \
    --save_model \
    --save_packed_model \
    --hessian_path /home/pengliang/.cache/huggingface/hub/models--relaxml--Hessians-Llama-2-7b-6144/snapshots/cafc59c036c6416ec2a9d5790752bec51297c197 \
    --inv_hessian_path /home/pengliang/Desktop/local_models/llama2_7b/inv_hessians \
    --ktol 1e-5 --kiter 100

