import numpy as np
from PIL import Image
import os

def gray(input: np.array) -> np.array:
    H, W = input.shape[:2]
    output = np.zeros((H, W), dtype=input.dtype)

    R = input[:, :, 0]
    G = input[:, :, 1]
    B = input[:, :, 2]
    output = 0.2126*R + 0.7152*G + 0.0722*B
    return output

def gaussianMask(radius: int) -> np.array:
    size = radius * 2 + 1
    center = size // 2

    i, j = np.indices((size, size))
    distance = np.ceil(np.sqrt((i - center)**2 + (j - center)**2))

    centerValue = 2**(radius + 1)
    mask = np.maximum(centerValue // 2**distance, 1)
    mask[center, center] = centerValue

    mask = mask / np.sum(mask)

    return mask

def oneMask(radius: int) -> np.array:
    size = radius * 2 + 1
    center = size // 2

    mask = np.ones((size, size))

    mask = mask / np.sum(mask)

    return mask

def addBlackBorder(inputArray: np.array, borderSize: int) -> np.array:
    H, W = inputArray.shape[:2]
    newH = H + 2 * borderSize
    newW = W + 2 * borderSize

    outputArray = np.zeros((newH, newW))
    outputArray[borderSize : borderSize + H, borderSize : borderSize + W] = inputArray

    return outputArray

def addWhiteBorder(inputArray: np.array, borderSize: int) -> np.array:
    H, W = inputArray.shape[:2]
    newH = H + 2 * borderSize
    newW = W + 2 * borderSize

    outputArray = np.ones((newH, newW)) * 255
    outputArray[borderSize : borderSize + H, borderSize : borderSize + W] = inputArray

    return outputArray

def deleteBorder(inputArray: np.array, borderSize: int) -> np.array:
    H, W = inputArray.shape[:2]
    outputArray = inputArray[borderSize : H - borderSize, borderSize : W - borderSize]

    return outputArray

def filter_gaussian(inputArray: np.array, radius: int) -> np.array:
    mask = gaussianMask(radius)

    H, W = inputArray.shape[:2]
    inputArray = addBlackBorder(inputArray, radius)
    outputArray = np.copy(inputArray)

    for i in range(radius, W + radius):
        for j in range(radius, H + radius):
            section = inputArray[j - radius : j + radius + 1, i - radius : i + radius + 1]
            section = section * mask
            outputArray[j, i] = np.sum(section)

    outputArray = deleteBorder(outputArray, radius)
    return outputArray

def filter_one(inputArray: np.array, radius: int) -> np.array:
    mask = oneMask(radius)

    H, W = inputArray.shape[:2]
    inputArray = addBlackBorder(inputArray, radius)
    outputArray = np.copy(inputArray)

    for i in range(radius, W + radius):
        for j in range(radius, H + radius):
            section = inputArray[j - radius : j + radius + 1, i - radius : i + radius + 1]
            section = section * mask
            outputArray[j, i] = np.sum(section)

    outputArray = deleteBorder(outputArray, radius)
    return outputArray

def main():
    inputFolder = "lab3/input/"
    outputFolder = "lab3/output/gaussian/"

    strArray = os.listdir(inputFolder)

    for fileName in strArray:
        inputStr = inputFolder + fileName

        inputImg = Image.open(inputStr).convert('RGB')
        inputArray = np.array(inputImg)

        greyArray = gray(inputArray)
        outputArray = filter_gaussian(greyArray, 3)

        outputImg = Image.fromarray(outputArray)
        outputImg = outputImg.convert("L")
        outputStr = outputFolder + fileName
        outputImg.save(outputStr)

        diffArray = np.abs(greyArray - outputArray)

        diffImg = Image.fromarray(diffArray)
        diffImg = diffImg.convert("L")
        diffStr = outputFolder + "/diff/" + fileName
        diffImg.save(diffStr)

if __name__ == "__main__":
    main()