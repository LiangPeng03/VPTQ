# VPTQ_OPT - 专为OPT模型设计的VPTQ量化工具

这是一个专门为OPT模型重构的VPTQ量化工具，保留了核心功能模块，包括：

## 项目结构

```
vptq_opt/
├── __init__.py
├── models/
│   └── opt.py          # OPT模型相关处理
├── layers/
│   └── vqlinear.py     # 向量量化线性层实现
├── utils/
│   ├── hessian.py      # Hessian矩阵处理工具
│   └── layer_utils.py  # 层处理工具
└── tools/
    ├── quantize_executer.py  # 量化执行器
    ├── layer_quantizer.py    # 层量化器
    └── run_quantization.py   # 主程序入口
```

## 核心模块说明

### 1. models/opt.py
- `get_opt()` - 加载原始OPT模型
- `quant_opt()` - 对OPT模型进行VPTQ量化
- `get_quantized_opt()` - 构建量化后的OPT模型
- `eval_opt()` - 评估OPT模型性能

### 2. layers/vqlinear.py
- `VQuantLinear` - 向量量化线性层类
- 实现了从码本和索引重构权重的机制

### 3. utils/hessian.py
- `load_hessian()` - 加载Hessian矩阵
- `load_inv_hessian()` - 加载逆Hessian矩阵

### 4. utils/layer_utils.py
- `find_layers()` - 查找模型中的指定层
- `replace_layer()` - 替换模型中的层

### 5. tools/quantize_executer.py
- `quantize_executer()` - 量化执行器主函数

### 6. tools/layer_quantizer.py
- `layer_quantizer()` - 层级量化器

## 使用方法

```bash
python vptq_opt/run_quantization.py \
    --model_name facebook/opt-125m \
    --hessian_path /path/to/hessian/files \
    --output_dir ./quantized_models
```

## 特点

1. **专为OPT模型设计** - 移除了对其他模型（如LLaMA、Qwen等）的支持，专注于OPT模型
2. **精简结构** - 仅保留核心量化流程所需模块
3. **易于扩展** - 清晰的模块划分便于后续功能扩展
4. **兼容性** - 保持与原VPTQ项目类似的接口设计

## 注意事项

1. 需要预先准备好Hessian矩阵文件
2. 当前版本主要面向单GPU量化场景
3. 某些高级功能（如多GPU支持）需要手动实现