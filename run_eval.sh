#!/bin/bash
CUDA_VISIBLE_DEVICES=0,1 /home/pengliang/glibc/glibc-2.32-install/lib/ld-linux-x86-64.so.2 \
  --library-path "/home/pengliang/glibc/glibc-2.32-install/lib:/usr/lib64:/usr/lib" \
  $(which python) pre_eval.py \
    --model_path /home/pengliang/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659 \