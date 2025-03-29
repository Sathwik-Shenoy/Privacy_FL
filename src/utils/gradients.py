import numpy as np

def sparse_gradient(gradient, threshold=1e-6):
    """Sparsify gradient by thresholding small values"""
    mask = np.abs(gradient) > threshold
    sparse_grad = gradient * mask
    return sparse_grad, np.mean(mask)

def reconstruct_gradient(sparse_grad, prev_grad, mask):
    return prev_grad + sparse_grad * mask