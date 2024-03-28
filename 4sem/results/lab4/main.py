import numpy as np
from PIL import Image

def grey(input: np.array) -> np.array:
    H, W = input.shape[:2]
    output = np.zeros((H, W), dtype=input.dtype)

    R = input[:, :, 0]
    G = input[:, :, 1]
    B = input[:, :, 2]
    output = 0.2126*R + 0.7152*G + 0.0722*B
    return output



def roberts(input: np.array) -> np.array:
    H, W = input.shape[:2]

    sobelX = np.array([[0, 0, 0],
                    [0, -1, 0],
                    [0, 0, 1]])

    sobelY = np.array([[0, 0, 0],
                    [0, 0, -1],
                    [0, 1, 0]])

    gradientX = np.zeros((H, W))
    gradientY = np.zeros((H, W))

    tempArray = np.zeros((H+2, W+2))
    tempArray[1:-1, 1:-1] = input

    for i in range(1, W+1):
        for j in range(1, H+1):
            gradientX[j-1, i-1] = np.abs(np.sum(sobelX * tempArray[j-1 : j+2, i-1 : i+2]))
            gradientY[j-1, i-1] = np.abs(np.sum(sobelY * tempArray[j-1 : j+2, i-1 : i+2]))

    gradient = (gradientX ** 2 + gradientY ** 2) ** (1/2)

    return gradient



def binarization(input: np.array, threshold: int) -> np.array:
    output = np.where(input > threshold, 255, 0)
    return output



def main():
    inputImg = Image.open("lab4/input/84_95.png")
    inputArray = np.array(inputImg)

    greyArray = grey(inputArray)
    gradientArray = roberts(greyArray)
    outputArray = binarization(gradientArray, 10)

    newImg = Image.fromarray(gradientArray)
    newImg.show()

if __name__ == "__main__":
    main()