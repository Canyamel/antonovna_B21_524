import numpy as np

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

def get_features_letter(input: np.array):
    H, W = input.shape[:2]
    weight = get_weight(input)
    #relative_weight = weight / (W * H)

    weight_1 = get_weight(input, 0, W // 2, 0, H // 2)
    relative_weight_1 = weight_1 / ((W // 2) * (H // 2))

    weight_2 = get_weight(input, W // 2, W, 0, H // 2)
    relative_weight_2 = weight_2 / ((W - W // 2) * (H // 2))

    weight_3 = get_weight(input, 0, W // 2, H // 2, H)
    relative_weight_3 = weight_3 / ((W // 2) * (H - H // 2))

    weight_4 = get_weight(input, W // 2, W, H // 2, H)
    relative_weight_4 = weight_4 / ((W - W // 2) * (H - H // 2))

    center_x, center_y = get_center(input, weight)
    relative_center_x = (center_x - 1) / (W - 1)
    relative_center_y = (center_y - 1) / (H - 1)

    inertia_x, inertia_y = get_inertia(input, center_x, center_y)
    relative_inertia_x = inertia_x / (W**2 * H**2)
    relative_inertia_y = inertia_y / (W**2 * H**2)

    return relative_weight_1, relative_weight_2, relative_weight_3, relative_weight_4, relative_center_x, relative_center_y, relative_inertia_x, relative_inertia_y