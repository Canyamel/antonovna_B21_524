import numpy as np
from PIL import Image
import os



def interpolation(inputArray: np.array, M: int) -> np.array:
    H, W = inputArray.shape[:2]
    newH = H * M
    newW = W * M

    outputArray = np.zeros((newH, newW, inputArray.shape[2]), dtype=inputArray.dtype)

    for y in range(newH):
        for x in range(newW):
            orig_y = y // M
            orig_x = x // M
            outputArray[y, x] = inputArray[orig_y, orig_x]

    return outputArray



def decimation(inputArray: np.array, N: int) -> np.array:
    H, W = inputArray.shape[:2]
    newH = H // N
    newW = W // N

    outputArray = np.zeros((newH, newW, inputArray.shape[2]), dtype=inputArray.dtype)

    for y in range(newH):
        for x in range(newW):
            orig_y = y * N
            orig_x = x * N
            outputArray[y, x] = inputArray[orig_y, orig_x]

    return outputArray



def two_step_resampling(inputArray: np.array, M: int, N: int) -> np.array:
    buf_array = interpolation(inputArray, M)
    outputArray = decimation(buf_array, N)
    return outputArray



def one_step_resampling(inputArray: np.array, M: int, N: int) -> np.array:
    H, W = inputArray.shape[:2]
    newH = H  * M // N
    newW = W * M // N

    outputArray = np.zeros((newH, newW, inputArray.shape[2]), dtype=inputArray.dtype)

    for y in range(newH):
        for x in range(newW):
            orig_y = y  * N // M
            orig_x = x * N // M
            outputArray[y, x] = inputArray[orig_y, orig_x]

    return outputArray


def main():
    currentFolder = os.path.dirname(os.path.abspath(__file__))
    inputFolder = os.path.join(currentFolder, 'input')
    outputFolder = os.path.join(currentFolder, 'output')

    inputImg = Image.open(os.path.join(input_folder, 'graphs.png')).convert('RGB')
    #inputImg.show()
    inputArray = np.array(inputImg)

    M, N = 2, 5

    for select in range(1, 5):
        match select:
            case 1:
                outputArray = interpolation(inputArray, M)

            case 2:
                outputArray = decimation(inputArray, N)

            case 3:
                outputArray = two_step_resampling(inputArray, M, N)

            case 4:
                outputArray = one_step_resampling(inputArray, M, N)

            case _:
                exit()
        outputImg = Image.fromarray(outputArray)
        #outputImg.show()

        outputPath = os.path.join(outputFolder, f'graphs_{select}.png')
        outputImg.save(outputPath)


if __name__ == '__main__':
    main()
