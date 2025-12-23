#!/usr/bin/env python3
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import json
import os.path as osp
import time
from dataclasses import dataclass, field
from typing import Optional

import torch
from transformers import AutoTokenizer, HfArgumentParser, set_seed

from vptq_opt.models.opt import eval_opt, get_opt, quant_opt


@dataclass
class VPTQArguments:
    model_name: str = field(default="facebook/opt-125m")
    seq_len: Optional[int] = field(default=None)
    quant_step: int = field(default=1)
    percdamp: float = field(default=0.01)
    blocksize: int = field(default=128)
    output_dir: str = field(default="outputs")
    seed: int = field(default=0)
    eval: bool = field(default=False)
    new_eval: bool = field(default=False)
    save_model: bool = field(default=False)
    save_packed_model: bool = field(default=False)
    disable_actorder: bool = field(default=False)
    hessian_path: Optional[str] = field(default=None)
    inv_hessian_path: Optional[str] = field(default=None)
    num_gpus: int = field(default=1)
    eval_nsamples: int = field(default=128)
    save_qlinear: bool = field(default=False)
    absorb_perm: bool = field(default=False)


@dataclass
class QuantizationArguments:
    vector_lens: tuple = field(default=(1, 8))
    num_centroids: tuple = field(default=(1, 256))
    num_res_centroids: tuple = field(default=(0, 0))
    npercent: float = field(default=0.0)
    group_size: int = field(default=8)
    group_num: int = field(default=1)
    kiter: int = field(default=20)
    ktol: float = field(default=1e-5)
    enable_norm: bool = field(default=False)
    norm_dim: int = field(default=0)
    enable_perm: bool = field(default=False)


if __name__ == "__main__":
    parser = HfArgumentParser((VPTQArguments, QuantizationArguments))
    args, quant_args = parser.parse_args_into_dataclasses()

    # set output folder based on time
    args.output_dir = osp.join(args.output_dir, time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()))

    set_seed(args.seed)

    # Load OPT model
    model = get_opt(args.model_name)
    
    # set sequence length
    if args.seq_len or model.seqlen is None:
        model.seqlen = args.seq_len
    print(f"model sequence length: {model.seqlen}")

    model.eval()

    tick = time.time()
    print(f'exp time: {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}')
    print(f"args: {args}")
    print(f"quant_args: {quant_args}")

    # Evaluate original model before quantization if requested
    if args.eval:
        print("Evaluating original model performance before quantization...")
        # 这里应该添加测试数据加载逻辑
        print("Original Model Evaluation skipped (no test data provided)")
        print("-" * 50)

    # Quantize OPT model
    model, quantizers = quant_opt(model, args, quant_args)

    # save model, not for inference
    if args.save_model:
        model_path = osp.join(args.output_dir, 'model/')
        model.save_pretrained(model_path)
        
        # save config 
        config_path = osp.join(args.output_dir, 'model/config.json')
        with open(config_path, 'w') as f:
            json.dump(model.config.to_dict(), f)

        print(f'save config to {config_path}')
        print(f'save model to {model_path}')
        tokenizer = AutoTokenizer.from_pretrained(f'{args.model_name}', legacy=False)

        tokenizer.save_pretrained(model_path)
        print(f"save tokenizer to {model_path}")

    model.eval()

    print("Quantization completed successfully!")
    print(f"Total time: {time.time() - tick:.2f}s")