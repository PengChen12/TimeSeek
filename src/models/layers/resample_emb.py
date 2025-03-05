import torch
import numpy as np
import torch.nn.functional as F

def resample_patchemb(old: torch.Tensor, new_patch_len: int):
    """重新采样patch embedding内核的权重到目标patch长度。

    这个函数通过近似地反转patch resizing的效果来重新采样patch embedding内核。
    重采样是为了匹配数据预处理管道中的patch resizing行为。

    参数:
      old: 需要重新采样的原始参数，应该是一个2D张量（patch_len, d_model）。
      new_patch_len: 目标patch长度，应该是一个整数。

    返回:
      重新采样后的patch embedding内核，返回为Tensor类型。
    """
    
    # 确保输入张量有2个维度
    assert old.dim() == 2, "输入张量应为2D (patch_len, d_model)"
    
    # 如果原始patch长度已经匹配新的patch长度，直接返回原始张量
    if old.size(0) == new_patch_len:
        return old

    # 定义一个辅助函数，使用PyTorch的双线性插值对张量进行缩放
    def resize(x_tensor, new_shape):
        return F.interpolate(x_tensor[None, None, :], size=new_shape, mode='linear', align_corners=False)[0, 0, :]
    
    # 定义一个辅助函数，生成缩放矩阵
    def get_resize_mat(old_shape, new_shape):
        mat = []
        
        # 遍历旧形状中的所有元素，生成基向量并进行缩放
        for i in range(old_shape):
            basis_vec = torch.zeros(old_shape, dtype=torch.float32)
            basis_vec[i] = 1.
            resized_basis_vec = resize(basis_vec, new_shape)
            mat.append(resized_basis_vec)
        
        # 将缩放后的基向量堆叠成矩阵并返回
        return torch.stack(mat).T

    # 生成旧形状和目标形状的缩放矩阵
    resize_mat = get_resize_mat(old.size(0), new_patch_len)
    
    # 计算缩放矩阵的伪逆
    resize_mat_pinv = torch.linalg.pinv(resize_mat.T)

    # 定义一个辅助函数，重新采样内核
    def resample_kernel(kernel):
        resampled_kernel = resize_mat_pinv @ kernel
        return resampled_kernel

    # 并行化处理每一列
    old_transposed = old.T  # 转置old以方便处理列
    resampled_kernels = torch.stack([resample_kernel(col) for col in old_transposed], dim=1)
    
    return resampled_kernels.T  # 转置回原始的patch_len x d_model格式
