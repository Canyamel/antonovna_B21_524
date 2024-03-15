import numpy as np
from PIL import Image
import os

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

    '''
    for i in range(W):
        sum = 0
        for j in range(H):
            sum += input[j, i]
            if i == 0:
                intImg[j, i] = sum
            else:
                intImg[j, i] = intImg[j, i-1] + sum

    for i in range(W):
        for j in range(H):
            w1 = max(i - r, 0)
            w2 = min(i + r, W - 1)
            h1 = max(j - r, 0)
            h2 = min(j + r, H - 1)

            count = (w2 - w1) * (h2 - h1)
            sum = intImg[h2, w2] - intImg[h1+1, w2] - intImg[h2, w1+1] + intImg[h1+1, w1+1]
            if input[j, i] * count <= sum * (1 - t):
                output[j, i] = 0
            else:
                output[j, i] = 255
    '''

    return output



def main():
    currentDir = os.path.dirname(os.path.abspath(__file__))
    inputFolder = os.path.join(currentDir, '1/input/')

    input_image = Image.open(os.path.join(inputFolder, 'im2.png')).convert('RGB')
    input_array = np.array(input_image)

    greyArray = grey(input_array)
    newArray = binarization(greyArray, 8, 0.15)

    newImg = Image.fromarray(newArray)
    newImg.show()



if __name__ == '__main__':
    main()