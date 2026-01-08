import math

# ==========================================
# 1. 输入参数配置区域 (请修改这里)
# ==========================================

# 模型层数 (例如 Llama-2-7b 是 32 层)
NUM_LAYERS = 32 

# 需要量化的矩阵形状列表 [(行, 列), (行, 列)...]
# 这里的例子是 Llama-2-7b 单层中常见的线性层形状:
# q, k, v, o_proj 通常是 4096 x 4096
# gate, up_proj 通常是 11008 x 4096
# down_proj 通常是 4096 x 11008
# MATRIX_SHAPES_PER_LAYER = [
#     (4096, 4096),   # q_proj
#     (4096, 4096),   # k_proj
#     (4096, 4096),   # v_proj
#     (4096, 4096),   # o_proj
#     (11008, 4096),  # gate_proj
#     (4096, 11008),  # down_proj
#     (11008, 4096)   # up_proj
# ]
MATRIX_SHAPES_PER_LAYER = [
    (768, 768),   # q_proj
    (768, 768),   # k_proj
    (768, 768),   # v_proj
    (768, 768),   # o_proj
    (3072, 768),  # gate_proj
    (768, 3072),  # down_proj
]

# 向量长度 (Vector Length)
VECTOR_LENGTH = 8

# 原始权重精度 (bits)，通常 FP16 为 16
PRECISION = 16

# 主码本大小 (Main Codebook Size), 例如 65536对应 16bit 索引
MAIN_CODEBOOK_SIZE = 65536

# 残差码本大小 (Residual Codebook Size)
# 如果不使用残差量化，请填 0 或 1
RESIDUAL_CODEBOOK_SIZE = 256

# ==========================================
# 2. 计算逻辑 (无需修改)
# ==========================================

def calculate_vptq_bits():
    print(f"{'='*20} VPTQ 计算报告 {'='*20}")
    
    # 1. 计算索引所需的位数 (Index Bits)
    # log2(主码本大小)
    main_idx_bits = math.ceil(math.log2(MAIN_CODEBOOK_SIZE))
    
    # log2(残差码本大小)
    res_idx_bits = 0
    if RESIDUAL_CODEBOOK_SIZE > 1:
        res_idx_bits = math.ceil(math.log2(RESIDUAL_CODEBOOK_SIZE))
    
    total_idx_bits_per_vector = main_idx_bits + res_idx_bits
    
    print(f"配置信息:")
    print(f"  - 向量长度: {VECTOR_LENGTH}")
    print(f"  - 主索引位宽: {main_idx_bits} bits (大小 {MAIN_CODEBOOK_SIZE})")
    print(f"  - 残差索引位宽: {res_idx_bits} bits (大小 {RESIDUAL_CODEBOOK_SIZE})")
    print(f"  - 每个向量总索引开销: {total_idx_bits_per_vector} bits")
    print(f"{'-'*55}")

    total_params = 0
    total_quantized_bits = 0
    
    # 遍历每一层，每一个矩阵进行累加
    for layer_idx in range(NUM_LAYERS):
        for shape in MATRIX_SHAPES_PER_LAYER:
            rows, cols = shape
            num_params = rows * cols
            
            # --- A. 索引占用 (Indices) ---
            # 向量个数 = 参数总量 / 向量长度
            num_vectors = num_params / VECTOR_LENGTH
            
            # 该矩阵的索引总大小 = 向量个数 * 每个向量的索引位宽
            matrix_index_bits = num_vectors * total_idx_bits_per_vector
            
            # --- B. 码本占用 (Codebooks / Overhead) ---
            # 码本本身存的是原始精度 (FP16) 的数值
            # 码本大小 = (主码本行数 + 残差码本行数) * 向量长度 * 原始精度
            
            # 主码本开销
            main_cb_bits = MAIN_CODEBOOK_SIZE * VECTOR_LENGTH * PRECISION
            
            # 残差码本开销 (如果有)
            res_cb_bits = 0
            if RESIDUAL_CODEBOOK_SIZE > 1:
                res_cb_bits = RESIDUAL_CODEBOOK_SIZE * VECTOR_LENGTH * PRECISION
            
            matrix_overhead_bits = main_cb_bits + res_cb_bits
            
            # --- 汇总 ---
            current_matrix_total_bits = matrix_index_bits + matrix_overhead_bits
            
            total_params += num_params
            total_quantized_bits += current_matrix_total_bits

    # 3. 计算结果
    original_bits = total_params * PRECISION
    compression_ratio = original_bits / total_quantized_bits
    effective_bitwidth = total_quantized_bits / total_params

    print(f"计算结果 (共 {NUM_LAYERS} 层):")
    print(f"  - 参数总量: {total_params:,}")
    print(f"  - 原始大小 (FP16): {original_bits / 8 / 1024 / 1024:.2f} MB")
    print(f"  - 量化后大小 (VPTQ): {total_quantized_bits / 8 / 1024 / 1024:.2f} MB")
    print(f"  - 压缩比: {compression_ratio:.2f}x")
    print(f"\n★ 实际量化比特数 (Effective Bitwidth): {effective_bitwidth:.4f} bits")
    print(f"{'='*55}")

if __name__ == "__main__":
    calculate_vptq_bits()