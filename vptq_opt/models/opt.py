# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import os
import time

import torch
import torch.nn as nn

from vptq_opt.layers.vqlinear import VQuantLinear
from vptq_opt.utils.layer_utils import find_layers, replace_layer
from vptq_opt.tools.quantize_executer import quantize_executer


name2hessian = {
    "self_attn.q_proj": "q",
    "self_attn.k_proj": "k", 
    "self_attn.v_proj": "v",
    "self_attn.out_proj": "o",
    "fc1": "up",
    "fc2": "down",
}


def get_opt(model_name, seqlen=None):
    """获取OPT模型"""
    def skip(*args, **kwargs):
        pass

    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    
    from transformers import OPTForCausalLM
    model = OPTForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16
    )

    if seqlen is not None:
        model.seqlen = seqlen
    return model


def quant_opt(model, args, quant_args, dev='cuda'):
    """量化OPT模型"""
    print('Starting VPTQ...')

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.decoder.layers

    model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev)
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
    if hasattr(model.model.decoder, 'project_out') and model.model.decoder.project_out:
        model.model.decoder.project_out = model.model.decoder.project_out.to(dev)
    if hasattr(model.model.decoder, 'project_in') and model.model.decoder.project_in:
        model.model.decoder.project_in = model.model.decoder.project_in.to(dev)
    model.model.decoder.final_layer_norm = model.model.decoder.final_layer_norm.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    print(f'model dtype: {dtype}')

    # save model to cpu
    model = model.cpu()

    layers[0] = layers[0].cpu()

    model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
    if hasattr(model.model.decoder, 'project_out') and model.model.decoder.project_out:
        model.model.decoder.project_out = model.model.decoder.project_out.cpu()
    if hasattr(model.model.decoder, 'project_in') and model.model.decoder.project_in:
        model.model.decoder.project_in = model.model.decoder.project_in.cpu()
    model.model.decoder.final_layer_norm = model.model.decoder.final_layer_norm.cpu()
    model.config.use_cache = use_cache

    torch.cuda.empty_cache()

    model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to('cpu')
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.to('cpu')
    if hasattr(model.model.decoder, 'project_out') and model.model.decoder.project_out:
        model.model.decoder.project_out = model.model.decoder.project_out.to('cpu')
    if hasattr(model.model.decoder, 'project_in') and model.model.decoder.project_in:
        model.model.decoder.project_in = model.model.decoder.project_in.to('cpu')
    model.model.decoder.final_layer_norm = model.model.decoder.final_layer_norm.to('cpu')
    layers[0] = layers[0].to('cpu')
    model = model.cpu()

    # multiple gpus VPTQ
    quantizers = {}
    layers = model.model.decoder.layers

    print(f'----quantization start ...---- {time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())}')

    # calculate task allocation
    total_layers = len(layers)
    num_gpus = min(args.num_gpus, total_layers)

    base_layers_per_gpu = total_layers // num_gpus
    remaining_layers = total_layers % num_gpus

    tasks = []
    current_layer_idx = 0

    # Distribute tasks to GPUs
    for gpu_idx in range(num_gpus):
        current_gpu_tasks = []

        # Calculate how many layers this GPU should handle
        layers_for_this_gpu = base_layers_per_gpu
        if gpu_idx < remaining_layers:
            layers_for_this_gpu += 1

        # Assign layers to this GPU
        for _ in range(layers_for_this_gpu):
            if current_layer_idx < total_layers:
                current_gpu_tasks.append((current_layer_idx, layers[current_layer_idx]))
                current_layer_idx += 1

        tasks.append(current_gpu_tasks)
        print(f'GPU {gpu_idx} assigned {len(current_gpu_tasks)} layers')

    # init multiprocessing
    layer_state_dicts = {}
    layer_qlinear_args = {}

    if args.num_gpus == 1:
        layer_state_dicts, layer_qlinear_args = quantize_executer(0, tasks[0], args, quant_args, None, None, name2hessian)
    else:
        raise NotImplementedError("Multi-GPU quantization not implemented in this simplified version")

    print(f'----quantization done ...---- {time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())}')

    # check if all layers are quantized
    if len(layer_state_dicts) != len(layers):
        print('Error: not all layers are quantized')
        exit(1)

    qmodel = get_quantized_opt(model.config._name_or_path, args.seq_len, layer_state_dicts, layer_qlinear_args)

    model = qmodel

    print(f'qmodel: {model}')

    torch.cuda.empty_cache()
    return model, quantizers


def get_quantized_opt(model_name, seqlen, layer_state_dicts, layer_qlinear_args):
    """获取量化后的OPT模型"""
    model = get_opt(model_name, seqlen=seqlen)
    dtype = next(iter(model.parameters())).dtype
    layers = model.model.decoder.layers

    for layer_idx, layer_state_dict in layer_state_dicts.items():
        layer = layers[layer_idx]
        ops = find_layers(layer)

        for name, op in ops.items():
            # init qlinear
            qlayer = VQuantLinear(
                **layer_qlinear_args[layer_idx][name],
                dtype=dtype,
            )
            module_name = name.split('.')[-1]
            replace_layer(layer, module_name, qlayer)

        # convert dtype
        for param_name, param in layer_state_dict.items():
            if layer_state_dict[param_name].dtype not in [
                dtype, torch.int64, torch.int32, torch.int16, torch.int8, torch.uint64, torch.uint32, torch.uint16,
                torch.uint8, torch.bool
            ]:
                layer_state_dict[param_name] = layer_state_dict[param_name].to(dtype)

        layers[layer_idx].load_state_dict(layer_state_dict)

    return model


@torch.no_grad()
def eval_opt(model, testenc, dev):
    """评估OPT模型"""
    current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    print(f'----Evaluating opt ...---- {current_time}')

    testenc = testenc.input_ids
    nsamples = testenc.numel() // model.seqlen

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.decoder.layers

    model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev)
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
    if hasattr(model.model.decoder, 'project_out') and model.model.decoder.project_out:
        model.model.decoder.project_out = model.model.decoder.project_out.to(dev)
    if hasattr(model.model.decoder, 'project_in') and model.model.decoder.project_in:
        model.model.decoder.project_in = model.model.decoder.project_in.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev)
    cache = {'i': 0, 'attention_mask': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs.get('position_ids', None)
            raise ValueError

    model.model.decoder.embed_positions = Catcher(model.model.decoder.embed_positions)
    for i in range(nsamples):
        batch = testenc[:, (i * model.seqlen):((i + 1) * model.seqlen)].to(dev)
        try:
            model(batch)
        except ValueError:
            pass
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.module

    torch.cuda.empty_cache()

    # Move layers to device one by one for evaluation
    for i in range(len(layers)):
        layer = layers[i].to(dev)
        
        # Process samples through this layer
        for j in range(nsamples):
            # Prepare attention mask
            attention_mask = cache['attention_mask'][j].to(dev) if cache['attention_mask'] is not None else None
            
            # Create proper attention mask format for OPT
            if attention_mask is not None:
                if attention_mask.dim() == 2:
                    attention_mask = attention_mask.unsqueeze(0).unsqueeze(0)
                    # Add head dimension
                    attention_mask = attention_mask.expand(-1, 1, -1, -1)  # Add head dimension
                
            # Forward pass through the layer
            output = layer(
                inps[j].unsqueeze(0), 
                attention_mask=attention_mask,
                output_attentions=False
            )
            
            # Update inputs for next layer
            inps[j] = output[0].squeeze(0)
        
        # Move layer back to CPU
        layers[i] = layer.cpu()
        
    # Process final layer norm
    if model.model.decoder.final_layer_norm is not None:
        model.model.decoder.final_layer_norm = model.model.decoder.final_layer_norm.to(dev)
        inps = model.model.decoder.final_layer_norm(inps)
        model.model.decoder.final_layer_norm = model.model.decoder.final_layer_norm.cpu()
    
    # Process final projection if it exists
    if hasattr(model.model.decoder, 'project_out') and model.model.decoder.project_out:
        model.model.decoder.project_out = model.model.decoder.project_out.to(dev)
        inps = model.model.decoder.project_out(inps)
        model.model.decoder.project_out = model.model.decoder.project_out.cpu()

    # Compute perplexity
    model.lm_head = model.lm_head.to(dev)
    
    loss_fct = nn.CrossEntropyLoss(reduction="none")
    shift_labels = testenc[:, model.seqlen:].contiguous()
    loss_all = []
    
    for i in range(nsamples):
        lm_logits = model.lm_head(inps[i]).contiguous()
        shift_logits = lm_logits[:-1, :]
        shift_labels_batch = shift_labels[i, :shift_logits.shape[0]]
        loss = loss_fct(shift_logits, shift_labels_batch)
        loss_all.append(loss)
    
    loss_all = torch.cat(loss_all, dim=0)
    ppl = torch.exp(loss_all.mean())
    
    model.lm_head = model.lm_head.cpu()
    model.config.use_cache = use_cache
    
    torch.cuda.empty_cache()
    
    return ppl.item()