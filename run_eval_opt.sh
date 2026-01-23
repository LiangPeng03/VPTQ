#!/bin/bash
CUDA_VISIBLE_DEVICES=0,1 /home/pengliang/glibc/glibc-2.32-install/lib/ld-linux-x86-64.so.2 \
  --library-path "/home/pengliang/glibc/glibc-2.32-install/lib:/usr/lib64:/usr/lib" \
  $(which python) eval_opt.py \
    --model_path /home/pengliang/Desktop/test_output/opt_125m/2026-01-22-07-50-21/packed_model \