#!/bin/bash
CUDA_VISIBLE_DEVICES=0,1 /home/pengliang/glibc/glibc-2.32-install/lib/ld-linux-x86-64.so.2 \
  --library-path "/home/pengliang/glibc/glibc-2.32-install/lib:/usr/lib64:/usr/lib" \
  $(which python) pre_eval.py \
    --model_path /home/pengliang/Desktop/test_output/Meta-Llama-2-7b/2025-11-09-06-16-18/packed_model \