import torch


def soft_permutation_labels(counts, sigma=10):
    """
    counts: (B, K) 
    returns: soft label permutation matrix (B, K, K)
    """
    B, K = counts.shape
    P = torch.zeros((B, K, K), device=counts.device)

    for b in range(B):
        for i in range(K):
            diffs = torch.exp(-((counts[b] - counts[b, i]) ** 2) / (2 * sigma**2)) 
            P[b, i] = diffs / diffs.sum()

    return P
