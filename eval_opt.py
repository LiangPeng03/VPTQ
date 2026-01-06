import time
import torch
import torch.nn as nn
from tqdm import tqdm

@torch.no_grad()
# 加速用的：不更新权重，不反向传播的时候用
def eval_opt(model, testenc, dev):
    current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(f'----Evaluating OPT ...---- {current_time}')

    testenc = testenc.input_ids
    nsamples = testenc.numel() // model.seqlen

    use_cache = model.config.use_cache
    model.config.use_cache = False
    
    # OPT模型的层结构与LLaMA不同
    layers = model.model.decoder.layers

    model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev)
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
    
    # 如果存在layernorm_embedding，则也需要移到设备上
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

    # OPT模型的最终归一化层在decoder中
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

#--——————————————————————————————————————————————


import numpy as np
# set seed for reproducibility
def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)


def get_wikitext2(nsamples, seed, seqlen, model):
    print('Loading dataset...')
    from datasets import load_dataset
    traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    print('Preprocessing dataset....')
    from transformers import AutoTokenizer
    # tokenizer 分词器，用于文本预处理，将文本转换为模型可接受的输入格式（数字队列）
    try:
        tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False, trust_remote_code=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True, trust_remote_code=True)
    # use_fast=False : 传统分词器，通常更稳定但速度较慢

    trainenc = tokenizer("\n\n".join(traindata['text']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')
    #  用双换行符连接文本数据，形成一个长字符串，然后进行编码（数字张量）

    import random
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc


def get_c4(nsamples, seed, seqlen, model):
    from datasets import load_dataset

    traindata = load_dataset(
        'allenai/c4', '', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train', num_proc=48
    )
    valdata = load_dataset(
        'allenai/c4',
        data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'},
        split='validation',
        num_proc=48
    )

    from transformers import AutoTokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False, trust_remote_code=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True, trust_remote_code=True)

    import random
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] > seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    import random
    random.seed(0)
    valenc = []
    for _ in range(nsamples):
        ##################  找一个足够长的样本  ##################
        while True:
            i = random.randint(0, len(valdata) - 1)
            tmp = tokenizer(valdata[i]['text'], return_tensors='pt')
            if tmp.input_ids.shape[1] >= seqlen:
                break
        ################
        i = random.randint(0, tmp.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        valenc.append(tmp.input_ids[:, i:j])
    valenc = torch.hstack(valenc)

    class TokenizerWrapper:

        def __init__(self, input_ids):
            self.input_ids = input_ids

    valenc = TokenizerWrapper(valenc)

    return trainloader, valenc


def get_c4_new(nsamples, seed, seqlen, model):
    from datasets import load_dataset

    traindata = load_dataset(
        'allenai/c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train', num_proc=48
    )
    valdata = load_dataset(
        'allenai/c4',
        data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'},
        split='validation',
        num_proc=48
    )

    from transformers import AutoTokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False, trust_remote_code=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True, trust_remote_code=True)

    import random
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] >= seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    valenc = tokenizer(' '.join(valdata[:1100]['text']), return_tensors='pt')
    valenc = valenc.input_ids[:, :(256 * seqlen)]

    class TokenizerWrapper:

        def __init__(self, input_ids):
            self.input_ids = input_ids

    valenc = TokenizerWrapper(valenc)

    return trainloader, valenc


def get_data_loader(name, nsamples=128, seed=0, seqlen=2048, model=''):
    if 'wikitext2' in name:
        return get_wikitext2(nsamples, seed, seqlen, model)
    if 'c4' in name:
        if 'new' in name:
            return get_c4_new(nsamples, seed, seqlen, model)
        return get_c4(nsamples, seed, seqlen, model)

#--——————————————————————————————————————————————

from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
import argparse

set_seed(0)


# 命令行参数解析：1.创建一个参数解析器对象 2.添加参数 3.解析参数，返回一个包含所有参数值的对象
parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, default="facebook/opt-125m")
args = parser.parse_args()

model = AutoModelForCausalLM.from_pretrained(args.model_path, local_files_only=True)

datasets = ['wikitext2', 'c4-new']
seqlens = [2048]
results = {}
model_name = args.model_path

for seqlen in seqlens:
    model.seqlen = seqlen
    for dataset in datasets:
        dataloader, testloader = get_data_loader(
            dataset, 
            seed=0, 
            model=model_name,
            seqlen=model.seqlen
        )
        
        print(f"Evaluating {dataset} with context length {seqlen}")
        
        if 'opt' in model_name.lower():
            ppl = eval_opt(model, testloader, 'cuda')
        elif 'llama' in model_name.lower() or 'mistral' in model_name.lower():
            raise NotImplementedError("LLaMA evaluation not implemented in this version")
        else:
            raise ValueError(f"Unsupported model: {model_name}")

        if f'ctx_{seqlen}' not in results:
            results[f'ctx_{seqlen}'] = {}
        results[f'ctx_{seqlen}'][dataset] = ppl
        print(f"PPL for {dataset}: {ppl}")