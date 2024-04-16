import numpy as np
from matplotlib import pyplot as plt

def grey(input: np.array) -> np.array:
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

def get_weight(input: np.array, W_start=None, W_end=None, H_start=None, H_end=None) -> int:
    if W_start is None:
        W_start = 0
    if H_start is None:
        H_start = 0
    if W_end is None:
        W_end = input.shape[1]
    if H_end is None:
        H_end = input.shape[0]

    weight = 0
    for x in range(W_start, W_end):
        for y in range(H_start, H_end):
            if input[y, x] == 1:
                weight += 1
    return weight

def get_center(input: np.array, weight: int):
    H, W = input.shape[:2]
    center_x, center_y = 0, 0
    for x in range(W):
        for y in range(H):
            if input[y, x] == 1:
                center_x += x
                center_y += y
    center_x = center_x / weight
    center_y = center_y / weight
    return center_x, center_y

def get_inertia(input: np.array, center_x: float, center_y: float):
    H, W = input.shape[:2]
    inertia_x, inertia_y = 0, 0
    for x in range(W):
        for y in range(H):
            if input[y, x] == 1:
                inertia_x += (x - center_x)**2
                inertia_y += (y - center_y)**2
    return inertia_x, inertia_y

def write_profile_x(input: np.array, number: int):
    H, W = input.shape[:2]
    profile = {
            'y': np.sum(input, axis=0),
            'x': np.arange(1, W + 1)
    }
    plt.bar(x=profile['x'], height=profile['y'])

    plt.xlim(0, W+1)
    plt.ylim(0, H+1)

    plt.savefig(f'lab5/output/x/{str(number)}.png')
    plt.clf()


def write_profile_y(input: np.array, number: int):
    H, W = input.shape[:2]
    profile = {
            'y': np.arange(1, H+1),
            'x': np.sum(input, axis=1)
    }
    plt.barh(y=profile['y'], width=profile['x'])

    plt.xlim(0, W+1)
    plt.ylim(H+1, 0)

    plt.savefig(f'lab5/output/y/{str(number)}.png')
    plt.clf()