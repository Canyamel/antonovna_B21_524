import numpy as np
from matplotlib import pyplot as plt

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

def write_profile_x(input: np.array):
    H, W = input.shape[:2]
    profile = {
            'y': np.sum(input, axis=0),
            'x': np.arange(1, W + 1)
    }
    plt.bar(x=profile['x'], height=profile['y'], width=0.7)

    plt.xlim(0, W+1)
    plt.ylim(0, H+1)

    plt.savefig(f'lab6/output/profile/profile_x.png')
    plt.show()
    plt.clf()


def write_profile_y(input: np.array):
    H, W = input.shape[:2]
    profile = {
            'y': np.arange(1, H+1),
            'x': np.sum(input, axis=1)
    }
    plt.barh(y=profile['y'], width=profile['x'], height=0.6)

    plt.xlim(0, W+1)
    plt.ylim(H+1, 0)

    plt.savefig(f'lab6/output/profile/profile_y.png')
    plt.show()
    plt.clf()

def get_borders_symbol(input: np.array) -> np.array:
    profile_x = input.sum(axis=0)
    borders = []

    i = 0
    while i < profile_x.shape[0]:
        if profile_x[i] != 0:
            x_start = i
            dist = 0
            while profile_x[i + dist] != 0:
                dist += 1
                if i + dist >= profile_x.shape[0]:
                    i += 1
                    break
            i += dist
            x_end = i - 1

            letter_array = input[:, x_start : x_end + 1]
            profile_y = letter_array.sum(axis=1)
            j = 0
            while j < profile_y.shape[0]:
                if profile_y[j] != 0:
                    y_start = j
                    break
                j += 1

            j = profile_y.shape[0] - 1
            while j >= 0:
                if profile_y[j] != 0:
                    y_end = j
                    break
                j -= 1

            borders.append(((x_start, y_start), (x_end, y_end)))
        i += 1

    return borders

def get_borders_line(input: np.array) -> np.array:
    profile_y = input.sum(axis=1)
    borders = []

    i = 0
    while i < profile_y.shape[0]:
        if profile_y[i] != 0:
            y_start = i
            dist = 0
            while profile_y[i + dist] != 0:
                dist += 1
                if i + dist >= profile_y.shape[0]:
                    i += 1
                    break
            i += dist
            y_end = i - 1

            letter_array = input[y_start : y_end + 1, :]
            profile_x = letter_array.sum(axis=0)
            j = 0
            while j < profile_x.shape[0]:
                if profile_x[j] != 0:
                    x_start = j
                    break
                j += 1

            j = profile_x.shape[0] - 1
            while j >= 0:
                if profile_x[j] != 0:
                    x_end = j
                    break
                j -= 1

            borders.append(((x_start, y_start), (x_end, y_end)))
        i += 1

    return borders

def get_borders_text(input: np.array) -> np.array:
    profile_x = input.sum(axis=0)
    profile_y = input.sum(axis=1)
    borders = []

    i = 0
    while i < profile_x.shape[0]:
        if profile_x[i] != 0:
            x_start = i
            break
        i += 1

    i = profile_x.shape[0] - 1
    while i >= 0:
        if profile_x[i] != 0:
            x_end = i
            break
        i -= 1

    i = 0
    while i < profile_y.shape[0]:
        if profile_y[i] != 0:
            y_start = i
            break
        i += 1

    i = profile_y.shape[0] - 1
    while i >= 0:
        if profile_y[i] != 0:
            y_end = i
            break
        i -= 1

    borders.append((x_start, y_start))
    borders.append((x_end, y_end))

    return borders