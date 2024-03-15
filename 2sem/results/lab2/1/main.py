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


def main():
    currentDir = os.path.dirname(os.path.abspath(__file__))
    inputFolder = os.path.join(currentDir, "input")
    outputFolder = os.path.join(currentDir, "output")

    strArray = os.listdir(inputFolder)

    for fileName in strArray:
        inputStr = os.path.join(inputFolder, fileName)

        inputImg = Image.open(inputStr).convert('RGB')
        inputArray = np.array(inputImg)

        greyArray = grey(inputArray)

        outputImg = Image.fromarray(greyArray)
        outputImg = outputImg.convert("L")
        outputStr = os.path.join(outputFolder, fileName)

        outputImg.save(outputStr)

if __name__ == '__main__':
    main()