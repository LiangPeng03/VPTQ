# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
# Adapted from https://github.com/Cornell-RelaxML/quip-sharp/blob/main/model/llama.py
# Modified to support OPT model quantization

import os
import time

import torch
import torch.multiprocessing as mp
import torch.nn as nn
from tqdm import tqdm

from vptq.layers.vqlinear import VQuantLinear
from vptq.quantize_executer import quantize_executer
from vptq.utils.layer_utils import find_layers, replace_layer


def get_opt(model_name, seqlen=None):
    """Get OPT model from transformers"""
    
    def skip(*args, **kwargs):
        pass

    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    
    from transformers import OPTForCausalLM
    model = OPTForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16
    )

    if seqlen is not None:
        model.seqlen = seqlen
    return model


def quant_opt(model, args, quant_args, dev='cuda'):
    """Quantize OPT model using VPTQ"""
    print('Starting VPTQ for OPT...')

    use_cache = model.config.use_cache
    model.config.use_cache = False
    # get decoder layers for OPT
    layers = model.model.decoder.layers

    # Move embedding layers to device
    model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev)
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
    
    # If layernorm_embedding exists, move it to device too
    if hasattr(model.model.decoder, 'layernorm_embedding'):
        model.model.decoder.layernorm_embedding = model.model.decoder.layernorm_embedding.to(dev)
        
    model.model.decoder.final_layer_norm = model.model.decoder.final_layer_norm.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    print(f'model dtype: {dtype}')

    # save model to cpu
    model = model.cpu()
    layers[0] = layers[0].cpu()

    model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
    if hasattr(model.model.decoder, 'layernorm_embedding'):
        model.model.decoder.layernorm_embedding = model.model.decoder.layernorm_embedding.cpu()
    model.model.decoder.final_layer_norm = model.model.decoder.final_layer_norm.cpu()
    model.config.use_cache = use_cache

    torch.cuda.empty_cache()

    model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to('cpu')
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.to('cpu')
    if hasattr(model.model.decoder, 'layernorm_embedding'):
        model.model.decoder.layernorm_embedding = model.model.decoder.layernorm_embedding.to('cpu')
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
            current_gpu_tasks.append((current_layer_idx, layers[current_layer_idx]))
            current_layer_idx += 1

        tasks.append(current_gpu_tasks)

    # print task allocation
    for gpu_idx in range(len(tasks)):
        task = [layer_idx for layer_idx, _ in tasks[gpu_idx]]
        print(f'gpu {gpu_idx} tasks: {task}')

    # init multiprocessing
    processes = []
    mq_manager = mp.get_context('spawn').Manager()
    input_queues = mq_manager.Queue()
    output_queues = mq_manager.Queue()

    # Define the OPT-specific name2hessian mapping
    opt_name2hessian = {
        'self_attn.q_proj': 'qkv',
        'self_attn.k_proj': 'qkv',
        'self_attn.v_proj': 'qkv',
        'self_attn.out_proj': 'o',
        'fc1': 'up',
        'fc2': 'down'
    }

    if args.num_gpus == 1:
        layer_state_dicts, layer_qlinear_args = quantize_executer(0, tasks[0], args, quant_args, None, None, name2hessian=opt_name2hessian)
    else:
        for gpu_idx in range(args.num_gpus):
            # we have to set CUDA_VISIBLE_DEVICES here
            # cuml only supports to run on GPU:0
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
            p = mp.Process(
                target=quantize_executer,
                args=(
                    gpu_idx,
                    tasks[gpu_idx],
                    args,
                    quant_args,
                    input_queues,
                    output_queues,
                    opt_name2hessian  # Pass the OPT-specific mapping
                )
            )

            p.start()
            processes.append(p)

        for p in processes:
            p.join()

    print(f'----quantization done ...---- {time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())}')

    # init quantized model
    model_name = model.model.config._name_or_path

    if args.num_gpus > 1:
        layer_state_dicts = {}
        layer_qlinear_args = {}

        # load save qlinear from files to avoid memory overflow
        if args.save_qlinear:
            for layer_idx in range(len(layers)):
                # load to cpu
                layer_state_dicts[layer_idx] = torch.load(
                    f'{args.output_dir}/qlinear_layer_state_{layer_idx}.pt', map_location='cpu',
                    weights_only=False
                )
                # bypass KeyError: torch.uint16
                for key, value in layer_state_dicts[layer_idx].items():
                    if "indices" in key:
                        layer_state_dicts[layer_idx][key] = value.view(torch.uint16)
                layer_qlinear_args[layer_idx] = torch.load(
                    f'{args.output_dir}/qlinear_args_{layer_idx}.pt', map_location='cpu',
                    weights_only=False
                )
        else:
            while not output_queues.empty():
                (gpu_id, layer_idx, _layer_state_dict, _layer_qlinear_args) = output_queues.get()
                layer_state_dicts[layer_idx] = _layer_state_dict
                layer_qlinear_args[layer_idx] = _layer_qlinear_args
                print(f'gpu {gpu_id} layer {layer_idx} quantized')

    # check if all layers are quantized
    if len(layer_state_dicts) != len(layers):
        print('Error: not all layers are quantized')
        exit(1)

    qmodel = get_quantized_opt(model_name, args.seq_len, layer_state_dicts, layer_qlinear_args)

    model = qmodel

    print(f'qmodel: {model}')

    torch.cuda.empty_cache()
    return model, quantizers


def get_quantized_opt(model_name, seqlen, layer_state_dicts, layer_qlinear_args):
    """Get quantized OPT model with loaded weights"""
    model = get_opt(model_name, seqlen=seqlen)
    dtype = next(iter(model.parameters())).dtype
    layers = model.model.decoder.layers

    for layer_idx, layer_state_dict in layer_state_dicts.items():
        # print(f'load quantized layer {layer_idx}')
        # print(f'layer_state_dict: {layer_state_dict.keys()}')
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
        # print(f'default dtype: {dtype}')
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
    """Evaluate OPT model"""
    current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    print(f'----Evaluating OPT ...---- {current_time}')

    testenc = testenc.input_ids
    nsamples = testenc.numel() // model.seqlen

    use_cache = model.config.use_cache
    model.config.use_cache = False
    
    # OPT model layers
    layers = model.model.decoder.layers

    model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev)
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
    
    # If layernorm_embedding exists, move to device
    if hasattr(model.model.decoder, 'layernorm_embedding'):
        model.model.decoder.layernorm_embedding = model.model.decoder.layernorm_embedding.to(dev)
        
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
            cache['attention_mask'] = kwargs.get('attention_mask', None)
            raise ValueError

    layers[0] = Catcher(layers[0])
    for i in range(nsamples):
        batch = testenc[:, (i * model.seqlen):((i + 1) * model.seqlen)].to(dev)
        try:
            model(input_ids=batch)
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
    if hasattr(model.model.decoder, 'layernorm_embedding'):
        model.model.decoder.layernorm_embedding = model.model.decoder.layernorm_embedding.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']

    for i in tqdm(range(len(layers))):
        layer = layers[i].to(dev)

        for j in range(nsamples):
            if attention_mask is not None:
                outs[j] = layer(inps[j].unsqueeze(0).to(dev), 
                              attention_mask=attention_mask)[0]
            else:
                outs[j] = layer(inps[j].unsqueeze(0).to(dev))[0]

        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    # OPT model final norm is in decoder
    if model.model.decoder.final_layer_norm is not None:
        model.model.decoder.final_layer_norm = model.model.decoder.final_layer_norm.to(dev)
    model.lm_head = model.lm_head.to(dev)

    testenc = testenc.to(dev)
    nlls = []
    for i in range(nsamples):
        hidden_states = inps[i].unsqueeze(0)
        if model.model.decoder.final_layer_norm is not None:
            hidden_states = model.model.decoder.final_layer_norm(hidden_states)
        lm_logits = model.lm_head(hidden_states)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = testenc[:, (i * model.seqlen):((i + 1) * model.seqlen)][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    print(ppl.item())

    model.config.use_cache = use_cache
    return ppl.item()