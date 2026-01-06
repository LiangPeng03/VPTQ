# VPTQ for OPT Models

This project extends the VPTQ (Vector Permutation Tensor Quantization) framework to support quantization of OPT (Open Pre-trained Transformer) models. The code is adapted from the original VPTQ implementation which was designed for LLaMA models.

## Overview

VPTQ-OP is a quantization framework specifically designed for OPT models. It implements vector permutation tensor quantization to achieve high compression ratios while maintaining model accuracy.

## Features

- Quantization of OPT models using VPTQ methodology
- Support for multi-GPU quantization
- Integration with Hugging Face Transformers
- Configurable quantization parameters
- Evaluation tools for quantized models

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/VPTQ_OPT.git
cd VPTQ_OPT

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Quantization

To quantize an OPT model:

```bash
bash run_quantization_opt.sh
```

Alternatively, you can run the quantization directly:

```bash
python run_vptq_opt.py \
    --model_name path/to/your/opt/model \
    --output_dir path/to/output \
    --vector_lens -1 8 \
    --group_num 1 \
    --num_centroids -1 65536 \
    --num_res_centroids -1 256 \
    --npercent 0 \
    --blocksize 128 \
    --new_eval \
    --seq_len 8192 \
    --kmeans_mode hessian \
    --num_gpus 1 \
    --enable_perm \
    --enable_norm \
    --save_model \
    --save_packed_model
```

### Evaluation

To evaluate a quantized OPT model:

```bash
python eval_opt.py --model_path path/to/your/quantized/model
```

## Configuration Options

- `--model_name`: Path to the pre-trained OPT model
- `--output_dir`: Directory to save the quantized model
- `--vector_lens`: Vector lengths for quantization (default: [-1, 8])
- `--group_num`: Number of groups for quantization (default: 1)
- `--num_centroids`: Number of centroids for primary quantization (default: -1)
- `--num_res_centroids`: Number of centroids for residual quantization (default: -1)
- `--npercent`: Percentage of outliers (default: 0)
- `--blocksize`: Block size for quantization (default: 128)
- `--seq_len`: Sequence length for processing (default: 2048)
- `--kmeans_mode`: K-means mode (default: 'hessian')
- `--num_gpus`: Number of GPUs to use (default: 1)
- `--enable_perm`: Enable permutation optimization
- `--enable_norm`: Enable normalization
- `--save_model`: Save the quantized model
- `--save_packed_model`: Save a packed version of the model

## Project Structure

```
VPTQ_OPT/
├── README.md
├── eval_opt.py          # OPT model evaluation utilities
├── run_vptq_opt.py      # Main script for OPT model quantization
├── run_quantization_opt.sh  # Script to run OPT quantization
├── vptq_opt/            # VPTQ implementation for OPT models
│   ├── __init__.py
│   └── opt.py           # OPT-specific quantization functions
└── requirements.txt     # Python dependencies
```

## Dependencies

- Python >= 3.8
- PyTorch >= 1.12
- Transformers
- Datasets
- Tqdm
- NumPy

## Results

The VPTQ framework for OPT models achieves competitive compression ratios while maintaining model quality. Quantized models can be used with minimal performance degradation compared to full-precision models.

## License

This project is licensed under the MIT License - see the LICENSE file for details.