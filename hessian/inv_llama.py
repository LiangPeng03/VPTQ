import torch
import os
from argparse import ArgumentParser

# ==================== 新增：强制正定化函数 ====================
def force_positive_definite(H, min_eig=1e-4, verbose=False):
    """
    通过特征值分解强制矩阵正定。
    原理：分解 H = V * diag(eigenvalues) * V^T
    将所有小于 min_eig 的特征值强制设为 min_eig，然后重构。
    """
    # 1. 确保是双精度，防止精度不够导致的特征值计算错误
    H_double = H.to(torch.float64)
    
    # 2. 对称矩阵特征值分解 (eigh 比 eig 更适合对称矩阵，且更稳定)
    # L 是特征值向量，V 是特征向量矩阵
    L, V = torch.linalg.eigh(H_double)
    
    if verbose:
        print(f"  [Debug] Min eigenvalue before fix: {L.min().item():.6e}")
        print(f"  [Debug] Max eigenvalue before fix: {L.max().item():.6e}")

    # 3. 修正特征值：将所有小于 min_eig 的值替换为 min_eig
    # 注意：如果 min_eig 设得太小（如 1e-6），在 float32 下可能依然会报错
    # 建议保持在 1e-4 或 1e-3 左右，这对量化精度影响很小
    L = torch.clamp(L, min=min_eig)
    
    # 4. 重构矩阵 H_new = V * diag(L) * V^T
    H_new = V @ (torch.diag(L) @ V.T)
    
    # 5. 转回原来的精度 (如果需要 float32)
    return H_new.to(H.dtype)

# ==================== 主逻辑 ====================

def load_hessian(hessian_path):
    print(f'load Hessian from {hessian_path}')
    H_data = torch.load(f'{hessian_path}', map_location='cpu') # 先加载到内存

    def flat_to_sym(V, N):
        A = torch.zeros(N, N, dtype=V.dtype, device=V.device)
        idxs = torch.tril_indices(N, N, device=V.device)
        A[idxs.unbind()] = V
        A[idxs[1, :], idxs[0, :]] = V
        return A

    H = flat_to_sym(H_data['flatH'], H_data['n'])
    mu = H_data['mu']
    n = H_data['n']
    
    # 注意：这里我们不在加载时做正则化，留到主循环做，以便控制
    H.add_(mu[None, :] * mu[:, None])
    
    return H, mu, n

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--load_hessian_dir', type=str, required=True)
    parser.add_argument('--store_inv_hessian_dir', type=str, required=True)
    parser.add_argument('--enable_perm', action='store_true')
    parser.add_argument('--sym', action='store_true')
    args = parser.parse_args()

    os.makedirs(args.store_inv_hessian_dir, exist_ok=True)

    hessian_files = [f for f in os.listdir(args.load_hessian_dir) if f.endswith('.pt')]

    # 建议使用 CUDA 进行特征值分解，CPU 会非常慢
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Processing on device: {dev}")

    for hessian_file in hessian_files:
        hessian_path = os.path.join(args.load_hessian_dir, hessian_file)
        
        try:
            hessian, mu, n = load_hessian(hessian_path)
            hessian = hessian.to(dev)
            
            # --- 数据清洗 ---
            # 1. 检查 NaN/Inf
            if not torch.isfinite(hessian).all():
                print(f"WARNING: Matrix {hessian_file} contains NaNs or Infs! Replacing with 0.")
                hessian = torch.nan_to_num(hessian, nan=0.0, posinf=1e4, neginf=-1e4)

            # 2. 处理死神经元 (对角线为 0 或极小值)
            diag_elements = torch.diag(hessian)
            # 任何对角线小于 1e-6 的，都视为死神经元，设为 1 以避免奇异
            dead_neuron_mask = diag_elements <= 1e-6 
            if dead_neuron_mask.any():
                print(f"  Fixing {dead_neuron_mask.sum()} dead neurons")
                hessian[dead_neuron_mask, dead_neuron_mask] = 1.0

            # --- 核心处理逻辑 ---
            inv_hessian = None
            
            # 尝试标准方法：少量阻尼
            # 这里我们给一个基础阻尼，防止本来就很好的矩阵也要去做特征值分解
            percdamp = 0.01
            damp = percdamp * torch.mean(torch.diag(hessian))
            diag_idx = torch.arange(n, device=dev)
            hessian_damped = hessian.clone()
            hessian_damped[diag_idx, diag_idx] += damp
            
            try:
                # 尝试 1: 标准 Cholesky
                L = torch.linalg.cholesky(hessian_damped)
                hessian_inv = torch.cholesky_inverse(L)
                inv_hessian = torch.linalg.cholesky(hessian_inv, upper=True)
                print(f"[{hessian_file}] Standard Cholesky Success")
            
            except RuntimeError:
                print(f"[{hessian_file}] Standard Cholesky Failed. Switching to EIGEN-REPAIR.")
                
                # 尝试 2: 终极大法 - 特征值重构
                # 使用原始矩阵进行修复（不带刚才加的阻尼，或者加上也可以）
                H_repaired = force_positive_definite(hessian, min_eig=1e-3, verbose=True)
                
                # 修复后必然可以 Cholesky
                # 注意：修复后的矩阵已经是正定的，通常不需要再加额外阻尼，但为了稳健可以微量加一点
                try:
                    L = torch.linalg.cholesky(H_repaired)
                    hessian_inv = torch.cholesky_inverse(L)
                    inv_hessian = torch.linalg.cholesky(hessian_inv, upper=True)
                    print(f"[{hessian_file}] Eigen-Repair Success")
                except RuntimeError as e:
                    # 如果这都失败了，说明矩阵数值范围有问题（极大极小值跨度太大）
                    print(f"[{hessian_file}] CRITICAL FAILURE even after repair: {e}")
                    # 最后的保底：直接返回单位矩阵（放弃这一层的二阶信息）
                    # 这会退化为 RTN (Round-to-Nearest)，但至少能让模型跑起来
                    print(f"[{hessian_file}] Fallback to Identity Matrix")
                    inv_hessian = torch.eye(n, device=dev)

            # --- 保存逻辑 (与原代码一致) ---
            # get permutation (based on original hessian diagonals)
            zero_idx = torch.diag(hessian) <= 1e-6
            perm = torch.argsort(torch.diag(hessian), descending=True)
            
            if args.enable_perm:
               # 注意：如果是 inv_hessian，这里的 perm 逻辑可能需要调整，
               # 但为了保持与 VPTQ 兼容，我们按原逻辑保存
               pass 

            save_path = os.path.join(args.store_inv_hessian_dir, hessian_file)
            
            def sym_to_flat(A):
                N = A.shape[-1]
                idxs = torch.tril_indices(N, N, device=A.device)
                return A[idxs.unbind()]
            
            if args.sym:
                flat_inv = sym_to_flat(inv_hessian)
                torch.save({'invH': flat_inv.to('cpu'), 'perm': perm.to('cpu'), 'zero_idx': zero_idx.to('cpu'), 'n': n}, save_path)
            else:
                torch.save({'invH': inv_hessian.to('cpu'), 'perm': perm.to('cpu'), 'zero_idx': zero_idx.to('cpu')}, save_path)
            
            print(f'Saved to {save_path}')
            
        except Exception as e:
            print(f"Error processing {hessian_file}: {e}")