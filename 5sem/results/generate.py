import numpy as np
from PIL import Image
import os
import csv
from PIL.ImageOps import invert
from help import grey, binarization, get_weight, get_center, get_inertia, write_profile_x, write_profile_y
from alphabet import GEORGIAN

def generate_table():
    with open('lab5/output/table.csv', 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        data = [
            "letter", "weight", "relative_weight", "weight_1", "relative_weight_1", "weight_2", "relative_weight_2", "weight_3",
            "relative_weight_3", "weight_4", "relative_weight_4", "center_x", "center_y", "relative_center_x", "relative_center_y",
            "inertia_x", "inertia_y", "relative_inertia_x", "relative_inertia_y"
        ]
        writer.writerows([data])

        currentDir = os.path.dirname(os.path.abspath(__file__))
        inputFolder = os.path.join(currentDir, "input")
        outputFolder = os.path.join(currentDir, "output/invert")

        strArray = os.listdir(inputFolder)

        i = 0
        for fileName in strArray:
            input_str = os.path.join(inputFolder, fileName)
            input_img = Image.open(input_str).convert('RGB')
            input_array = np.array(input_img)

            H, W = input_array.shape[:2]
            input_grey = grey(input_array)
            input_bin = binarization(input_grey, 5, 0.15)
            input_bin[input_bin == 0] = 1
            input_bin[input_bin == 255] = 0

            weight = get_weight(input_bin)
            relative_weight = weight / (W * H)

            weight_1 = get_weight(input_bin, 0, W // 2, 0, H // 2)
            relative_weight_1 = weight / ((W // 2) * (H // 2))

            weight_2 = get_weight(input_bin, W // 2, W, 0, H // 2)
            relative_weight_2 = weight / ((W - W // 2) * (H // 2))

            weight_5 = get_weight(input_bin, 0, W // 2, H // 2, H)
            relative_weight_5 = weight / ((W // 2) * (H - H // 2))

            weight_4 = get_weight(input_bin, W // 2, W, H // 2, H)
            relative_weight_4 = weight / ((W - W // 2) * (H - H // 2))

            center_x, center_y = get_center(input_bin, weight)
            relative_center_x = (center_x - 1) / (W - 1)
            relative_center_y = (center_y - 1) / (H - 1)

            inertia_x, inertia_y = get_inertia(input_bin, center_x, center_y)
            relative_inertia_x = inertia_x / (W**2 * H**2)
            relative_inertia_y = inertia_y / (W**2 * H**2)

            data = [
                GEORGIAN[i], weight, round(relative_weight, 5), round(weight_1, 5), round(relative_weight_1, 5), round(weight_2, 5), round(relative_weight_2, 5),
                round(weight_5, 5), round(relative_weight_5, 5), round(weight_4, 5), round(relative_weight_4, 5), round(center_x, 5), round(center_y, 5),
                round(relative_center_x, 5), round(relative_center_y, 5),
                round(inertia_x, 5), round(inertia_y, 5), round(relative_inertia_x, 5), round(relative_inertia_y, 5)
            ]
            writer.writerow(data)
            i += 1
            write_profile_x(input_bin, i)
            write_profile_y(input_bin, i)

            outputStr = os.path.join(outputFolder, fileName)
            invert_img = invert(input_img)
            invert_img.save(outputStr)