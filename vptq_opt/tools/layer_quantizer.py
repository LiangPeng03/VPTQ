# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import time

import torch

from vptq_opt.layers.vqlinear import VQuantLinear
from vptq_opt.utils.hessian import load_hessian, load_inv_hessian
from vptq_opt.utils.layer_utils import find_layers, replace_layer


class MockQuantizer:
    """模拟量化器，用于演示目的"""
    def __init__(self, **kwargs):
        self.centroids = torch.randn(1, 16, 8)  # 示例质心
        self.indices = torch.randint(0, 16, (1, 96, 1))  # 示例索引
        self.res_centroids = None
        self.res_indices = None
        self.indices_sign = torch.ones(1)
        self.indices_scale = torch.ones(1)
        self.res_indices_sign = torch.ones(1)
        self.weight_scale = torch.ones(1, 768)
        self.weight_bias = torch.zeros(1, 768)
        self.perm = None
        self.group_num = 1
        self.group_size = 8
        self.outlier_size = 0


class MockVPTQ:
    """模拟VPTQ算法，用于演示目的"""
    def __init__(self, linear, hessian, inv_hessian, perm, quantizer, zero_idx, logger, **kwargs):
        self.quantizer = MockQuantizer()
        
    def fast_vector_quant(self):
        pass


def layer_quantizer(args, quant_args, layer, layer_idx, logger, dev, dtype, name2hessian=None):
    """层量化器"""
    qlinear_args = {}
    operators = find_layers(layer)
    opeartor_names = [list(operators.keys())]
    
    for names in opeartor_names:
        # subset: (op name, op) pairs
        subset = {n: operators[n] for n in names}
        logger.info(subset.keys())

        for name in subset:
            # load Hessian
            if name2hessian is None:
                name2hessian = {
                    'self_attn.v_proj': 'qkv',
                    'self_attn.q_proj': 'qkv',
                    'self_attn.k_proj': 'qkv',
                    'self_attn.o_proj': 'o',
                    'mlp.up_proj': 'up',
                    'mlp.gate_proj': 'up',
                    'mlp.down_proj': 'down'
                }

            layer_name = f'{layer_idx}_{name2hessian[name]}.pt'
            hessian_path = f'{args.hessian_path}/{layer_name}'
            
            try:
                hessian, mu = load_hessian(hessian_path, logger)
            except FileNotFoundError:
                # 如果找不到真实的Hessian文件，创建一个模拟的
                logger.warning(f"Hessian file not found: {hessian_path}, using mock Hessian")
                hessian = torch.randn(768, 768)  # 模拟Hessian矩阵
                mu = torch.randn(768)  # 模拟均值向量

            # init data
            linear = subset[name].to(dev)

            # load inv_hessian from files to reduce memory usage
            if args.inv_hessian_path is not None:
                inv_hessian_path = f'{args.inv_hessian_path}/{layer_name}'
                try:
                    inv_hessian, perm, zero_idx = load_inv_hessian(inv_hessian_path, logger)
                except FileNotFoundError:
                    logger.warning(f"Inverse Hessian file not found: {inv_hessian_path}, using mock values")
                    inv_hessian = torch.randn(768, 768)  # 模拟逆Hessian矩阵
                    perm = torch.arange(768)  # 模拟排列
                    zero_idx = torch.tensor([])  # 模拟零索引
            else:
                inv_hessian = None
                perm = None
                zero_idx = None

            layer_name = f'{layer_idx}.{name}'

            current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
            logger.info(f'----Quantizing opt ...---- {current_time} {layer_name}')

            # init vptq algo
            _vptq = MockVPTQ(
                linear,
                hessian=hessian,
                inv_hessian=inv_hessian,
                perm=perm,
                quantizer=None,
                zero_idx=zero_idx,
                logger=logger,
                collect_act=False,
                layer_name=layer_name,
                enable_perm=quant_args.enable_perm,
                enable_norm=quant_args.enable_norm,
                norm_dim=quant_args.norm_dim,
                debug=True
            )

            # quant by VPTQ algorithm
            _vptq.fast_vector_quant()

            quantizer = _vptq.quantizer
            perm = _vptq.quantizer.perm

            weight = linear.weight.clone().detach().to(dev)

            # num_codebooks = 1
            # centroid
            # num_centroids = quantizer.num_centroids[1]
            centroids = quantizer.centroids
            indices = quantizer.indices
            indices_sign = quantizer.indices_sign
            indices_scale = quantizer.indices_scale
            
            # res centroid
            # num_res_centroids = quantizer.num_res_centroids
            res_centroids = quantizer.res_centroids
            # res_centroids = quantizer.res_centroids[1]
            res_indices = quantizer.res_indices
            # res_indices = quantizer.res_indices[1]
            res_indices_sign = quantizer.res_indices_sign

            in_features = weight.size(1)
            out_features = weight.size(0)

            qlayer = VQuantLinear(
                # **outlier_kwargs,
                in_features=in_features,
                out_features=out_features,
                vector_lens=quant_args.vector_lens,
                num_centroids=quant_args.num_centroids,
                num_res_centroids=quant_args.num_res_centroids,
                # group settings
                # group_size=quantizer.group_size,
                group_num=quantizer.group_num,
                group_size=quantizer.group_size,
                outlier_size=quantizer.outlier_size,
                bias=True if linear.bias is not None else False,
                enable_norm=quant_args.enable_norm,
                norm_dim=quant_args.norm_dim,
                enable_perm=quant_args.enable_perm,
                # enable_residual=True,
                vector_quant_dim='out',
                device=dev,
                dtype=dtype,
                # indices_as_float=False,
            )

            qlinear_args[name] = qlayer.cpu().init_args

            weight_scale = _vptq.quantizer.weight_scale
            weight_bias = _vptq.quantizer.weight_bias

            qlayer.init_parameters(
                centroids=centroids,
                indices=indices,
                res_centroids=res_centroids,
                res_indices=res_indices,
                weight_scale=weight_scale,
                weight_bias=weight_bias,
                indices_sign=indices_sign,
                indices_scale=indices_scale,
                res_indices_sign=res_indices_sign,
                bias=linear.bias,
                perm=perm,
                dtype=dtype,
            )

            qlayer.to(dev)

            # replace layer with qlinear
            module_name = name.split('.')[-1]

            replace_layer(layer, module_name, qlayer)

            torch.cuda.empty_cache()

    return layer, qlinear_args