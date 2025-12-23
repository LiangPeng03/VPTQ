# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import math
from typing import Tuple

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter


class VQuantLinear(nn.Module):
    """向量量化线性层"""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        # vector length
        # [outlier vector length, vector length]
        vector_lens: Tuple[int, int],
        # centroids
        # [outlier num centroids, num centroids]
        num_centroids: Tuple[int, int],
        # [outlier num res centroids, num res centroids]
        num_res_centroids: Tuple[int, int],
        group_num: int,
        group_size: int,
        outlier_size: int,
        enable_norm: bool = False,
        norm_dim: int = 0,
        enable_perm: bool = False,
        is_indice_packed: bool = False,
        # configuration
        bias: bool = False,
        vector_quant_dim: str = "out",
        device=None,
        dtype=None,
        debug=False,
        # deprecated
        indices_as_float=None,
        enable_proxy_error=False,
    ):
        super().__init__()

        # get init args
        self.init_args = {
            "in_features": in_features,
            "out_features": out_features,
            "vector_lens": vector_lens,
            "num_centroids": num_centroids,
            "num_res_centroids": num_res_centroids,
            "group_num": group_num,
            "group_size": group_size,
            "outlier_size": outlier_size,
            "enable_norm": enable_norm,
            "norm_dim": norm_dim,
            "enable_perm": enable_perm,
            "norm_dim": norm_dim,
            "bias": bias,
            "is_indice_packed": is_indice_packed,
            "indices_as_float": indices_as_float,
            "enable_proxy_error": enable_proxy_error,
        }

        factory_kwargs = {"device": device, "dtype": dtype}

        self.in_features = in_features
        self.out_features = out_features
        if bias:
            self.bias = Parameter(torch.empty(self.out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)

        # set configuration
        self.debug = debug

        # to reduce index size and bypass nccl check
        self.is_indice_packed = is_indice_packed
        # TODO: FIX magic number
        self.vector_len = vector_lens[1]
        self.num_centroids = num_centroids[1]
        self.num_res_centroids = num_res_centroids[1]

        self.group_num = group_num
        self.num_codebooks = self.group_num
        self.outlier_size = outlier_size

        # set vector quantization parameters
        # current implementation only supports vector_len = 'out'
        assert vector_quant_dim in ["in", "out"]
        assert vector_quant_dim == "out"

        self.vector_quant_dim = vector_quant_dim
        # padding for vector quantization
        if self.vector_quant_dim == "in":
            assert True, "Not implemented"
        elif self.vector_quant_dim == "out":
            self.padding_size = (
                math.ceil(self.out_features / self.vector_len) * self.vector_len - self.out_features
            )

        # register parameters
        # centroids: [num_codebooks, num_centroids, vector_len]
        self.centroids = nn.Parameter(
            torch.empty((self.num_codebooks, self.num_centroids, self.vector_len), **factory_kwargs),
            requires_grad=False,
        )
        # indices: [num_codebooks, out_feature // vector_len, 1]
        if self.is_indice_packed:
            # when is_indice_packed is True,
            # indices is a packed tensor with
            # shape [num_codebooks, (out_feature // vector_len) // pack_factor, 1]
            # where pack_factor is determined by the data type of indices
            # e.g. torch.uint8 can pack 2 uint4 indices
            self.indices = nn.Parameter(
                torch.empty(
                    (
                        self.num_codebooks,
                        math.ceil(self.out_features / self.vector_len),
                        1,
                    ),
                    dtype=torch.uint16,
                    device=device,
                ),
                requires_grad=False,
            )
        else:
            self.indices = nn.Parameter(
                torch.empty((self.num_codebooks, self.out_features // self.vector_len, 1), dtype=torch.int64, **factory_kwargs),
                requires_grad=False,
            )

        # residual centroids
        if self.num_res_centroids > 0:
            # res_centroids: [num_codebooks, num_res_centroids, vector_len]
            self.res_centroids = nn.Parameter(
                torch.empty((self.num_codebooks, self.num_res_centroids, self.vector_len), **factory_kwargs),
                requires_grad=False,
            )
            # res_indices: [num_codebooks, out_feature // vector_len, 1]
            self.res_indices = nn.Parameter(
                torch.empty((self.num_codebooks, self.out_features // self.vector_len, 1), dtype=torch.int64, **factory_kwargs),
                requires_grad=False,
            )
        else:
            self.res_centroids = None
            self.res_indices = None

        # scale and bias for normalization
        if enable_norm:
            # weight_scale: [1, out_features]
            # weight_bias: [1, out_features]
            self.weight_scale = nn.Parameter(torch.empty((1, self.out_features), **factory_kwargs), requires_grad=False)
            self.weight_bias = nn.Parameter(torch.empty((1, self.out_features), **factory_kwargs), requires_grad=False)
        else:
            self.weight_scale = None
            self.weight_bias = None

        # permutation
        if enable_perm:
            # perm: [out_features]
            self.perm = nn.Parameter(torch.empty((self.out_features,), dtype=torch.int64, **factory_kwargs), requires_grad=False)
        else:
            self.perm = None

        # indices sign
        self.indices_sign = nn.Parameter(torch.empty((1,), **factory_kwargs), requires_grad=False)

        # indices scale
        self.indices_scale = nn.Parameter(torch.empty((1,), **factory_kwargs), requires_grad=False)

        # res indices sign
        self.res_indices_sign = nn.Parameter(torch.empty((1,), **factory_kwargs), requires_grad=False)

    def init_parameters(
        self,
        centroids,
        indices,
        res_centroids=None,
        res_indices=None,
        weight_scale=None,
        weight_bias=None,
        indices_sign=None,
        indices_scale=None,
        res_indices_sign=None,
        bias=None,
        perm=None,
        dtype=None,
    ):
        """初始化参数"""
        device = self.centroids.device
        
        # 初始化质心
        self.centroids.data = centroids.to(device)
        
        # 初始化索引
        self.indices.data = indices.to(device)
        
        # 初始化残差质心和索引
        if res_centroids is not None and self.res_centroids is not None:
            self.res_centroids.data = res_centroids.to(device)
            self.res_indices.data = res_indices.to(device)
        
        # 初始化权重缩放和偏移
        if weight_scale is not None and self.weight_scale is not None:
            self.weight_scale.data = weight_scale.to(device)
            self.weight_bias.data = weight_bias.to(device)
        
        # 初始化符号和缩放
        if indices_sign is not None:
            self.indices_sign.data = indices_sign.to(device)
        if indices_scale is not None:
            self.indices_scale.data = indices_scale.to(device)
        if res_indices_sign is not None and self.res_indices_sign is not None:
            self.res_indices_sign.data = res_indices_sign.to(device)
            
        # 初始化偏置
        if bias is not None and self.bias is not None:
            self.bias.data = bias.to(device)
            
        # 初始化排列
        if perm is not None and self.perm is not None:
            self.perm.data = perm.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        # 获取输入张量的形状
        x_shape = x.shape
        # 将输入展平为二维张量
        x = x.view(-1, x_shape[-1])
        
        # 获取权重矩阵
        weight = self.get_weight()
        
        # 执行线性变换
        out = F.linear(x, weight, self.bias)
        
        # 恢复输出形状
        out = out.view(*x_shape[:-1], out.shape[-1])
        return out

    def get_weight(self):
        """获取权重矩阵"""
        # 这里应该实现从码本和索引重构权重的逻辑
        # 简化版本，实际实现应该更复杂
        weight = torch.zeros(self.out_features, self.in_features, device=self.centroids.device)
        return weight