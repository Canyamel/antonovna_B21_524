import numpy as np

def gray(input: np.array) -> np.array:
    H, W = input.shape[:2]
    output = np.zeros((H, W), dtype=input.dtype)

    R = input[:, :, 0]
    G = input[:, :, 1]
    B = input[:, :, 2]
    output = 0.2126*R + 0.7152*G + 0.0722*B

    return output

def binarization(input: np.array, d: int, t: float) -> np.array:
    H, W = input.shape[:2]
    output = np.zeros((H, W), dtype=input.dtype)
    r = W  // (d * 2)

    intImg = input.cumsum(axis=0).cumsum(axis=1)

    i, j = np.meshgrid(np.arange(W), np.arange(H))
    w1 = np.maximum(i - r, 0)
    w2 = np.minimum(i + r, W - 1)
    h1 = np.maximum(j - r, 0)
    h2 = np.minimum(j + r, H - 1)

    count = (w2 - w1) * (h2 - h1)
    sum_val = intImg[h2, w2] - intImg[h1 + 1, w2] - intImg[h2, w1 + 1] + intImg[h1 + 1, w1 + 1]

    mask = input * count <= sum_val * (1 - t)
    output[mask] = 0
    output[~mask] = 255

    return output