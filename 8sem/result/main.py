import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from numpy import log
import os

def gray(input_img):
    W = input_img.size[0]
    H = input_img.size[1]

    input_array = np.array(input_img)
    output_array = np.zeros((H, W), dtype=input_array.dtype)

    R = input_array[:, :, 0]
    G = input_array[:, :, 1]
    B = input_array[:, :, 2]
    output_array = 0.2126*R + 0.7152*G + 0.0722*B

    output_img = Image.fromarray(output_array)
    output_img = output_img.convert("L")

    return output_img

def get_haralick(input_img, d=2):
    input_array = np.array(input_img)

    matrix = np.zeros((256, 256))
    H, W = input_array.shape[:2]
    for i in range(d, W - d):
        for j in range(d, H - d):
            matrix[int(input_array[j - d, i - d]), int(input_array[j, i])] += 1
            matrix[int(input_array[j + d, i - d]), int(input_array[j, i])] += 1
            matrix[int(input_array[j - d, i + d]), int(input_array[j, i])] += 1
            matrix[int(input_array[j + d, i + d]), int(input_array[j, i])] += 1

    for i in range(256):
        m = np.array(matrix[i])
        m[m == 0] = 1
        matrix[i] = log(m)

    matrix = matrix * 256 / np.max(matrix)

    return Image.fromarray(matrix), matrix

def get_Pj(i, matrix):
    Pj = 0
    for j in range(matrix.shape[1]):
        Pj += matrix[i, j]
    return Pj

def get_features_2(matrix):
    H, W = matrix.shape[:2]

    av = 0
    for i in range(H):
        av += i * get_Pj(i, matrix)

    d = 0
    for i in range(H):
            d += (i - av) ** 2 * get_Pj(i, matrix)
    d = d**(1/2)

    return av, d

def get_features(matrix):
    H, W = matrix.shape[:2]

    av = np.sum(matrix) / (W * H)

    d = 0
    for i in range(0, H):
        for j in range(0, W):
            d += (matrix[i, j] - av) ** 2
    d = (d / (W * H))**(1/2)

    return av, d

def get_hist(image, i, constrast):
    matrix = np.asarray(image)
    shape = np.reshape(matrix, (1, -1))
    plt.figure()
    plt.hist(shape[0], bins=256)
    if constrast:
        plt.savefig("lab8/output/img" + str(i) + "_hist.png")
    else:
        plt.savefig("lab8/output/img" + str(i) + "_contrast_hist.png")

def make_contrast_equalize_histogram(img_gray):
    img_array = np.array(img_gray)
    img_array = img_array.astype(int)

    histogram = np.bincount(img_array.flatten(), minlength=256)
    pmf = histogram / np.sum(histogram)
    cdf = np.cumsum(pmf)

    cdf = cdf * (256 - 1) // 1
    cdf = cdf.astype(int)

    H, W = img_array.shape[:2]
    for i in range(W):
        for j in range(H):
            img_array[j][i] = cdf[img_array[j][i]]

    output_img = Image.fromarray(img_array)
    output_img = output_img.convert("L")

    return output_img

def main():
    str_array = os.listdir("lab8/input/")
    i = 1
    for file_name in str_array:
        img_path = "lab8/input/" + file_name
        img = Image.open(img_path)
        img_gray = gray(img)
        img_gray.save("lab8/output/img" + str(i) + "_gray.png")

        contrast_img = make_contrast_equalize_histogram(img_gray)
        contrast_img.save("lab8/output/img" + str(i) + "_contrast.png")

        get_hist(img_gray, i, True)

        get_hist(contrast_img, i, False)

        haralick_img, haralick_matrix = get_haralick(img_gray)
        haralick_img = haralick_img.convert("L")
        haralick_img.save("lab8/output/img" + str(i) + "_haralick.png")

        contrast_haralick_img, contrast_haralick_matrix = get_haralick(contrast_img)
        contrast_haralick_img = contrast_haralick_img.convert("L")
        contrast_haralick_img.save("lab8/output/img" + str(i) + "_contrast_haralick.png")

        av, d = get_features(haralick_matrix)
        contrast_av, contrast_d = get_features(contrast_haralick_matrix)

        print("=====================")
        print(f"img = {i}")
        print(f"AV = {av}")
        print(f"D = {av}")
        print(f"contrast_AV = {contrast_av}")
        print(f"contrast_D = {contrast_d}")
        print("=====================")
        i += 1

if __name__ == "__main__":
    main()