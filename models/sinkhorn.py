import torch


def build_log_alpha(scores, tau=0.1):
    """
    scores: (B, K)  每張圖的一維排序分數
    return: log_alpha: (B, K, K)  對應每張圖放在每個順位的相對偏好
    """
    B, K = scores.shape
    device = scores.device

    # scores_s: (B, K, 1)
    scores_s = scores.view(B, K, 1)

    # 目標順位 (0,1,...,K-1)
    positions = torch.arange(K, device=device).float().view(1, 1, K)

    # 越接近某個順位，cost 越小 → log_alpha 越大
    log_alpha = -torch.abs(scores_s - positions) / tau
    return log_alpha


def sinkhorn(log_alpha, n_iters=20, eps=1e-6):
    """
    log_alpha: (B, K, K)
    return: P: (B, K, K)  近似 permutation 的 doubly-stochastic matrix
    """
    # exp to get positive
    Q = torch.exp(log_alpha)

    for _ in range(n_iters):
        # row normalize
        Q = Q / (Q.sum(dim=2, keepdim=True) + eps)
        # col normalize
        Q = Q / (Q.sum(dim=1, keepdim=True) + eps)

    return Q
