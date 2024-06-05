import numpy as np
from PIL import Image
import os
import csv
from PIL.ImageOps import invert
from help import gray, binarization, get_weight, get_center, get_inertia, write_profile_x, write_profile_y

def generate_table(alphabet):
    with open('lab5/output/table.csv', 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        data = [
            "letter", "weight", "relative_weight", "weight_1", "relative_weight_1", "weight_2", "relative_weight_2", "weight_3",
            "relative_weight_3", "weight_4", "relative_weight_4", "center_x", "center_y", "relative_center_x", "relative_center_y",
            "inertia_x", "inertia_y", "relative_inertia_x", "relative_inertia_y"
        ]
        writer.writerows([data])

        current_dir = os.path.dirname(os.path.abspath(__file__))
        input_folder = os.path.join(current_dir, "input")
        output_folder = os.path.join(current_dir, "output/invert")

        strArray = os.listdir(input_folder)
        for i in range(1, len(alphabet)+1):
            file_name = str(i) + '.png'
            input_str = os.path.join(input_folder, file_name)
            input_img = Image.open(input_str).convert('RGB')
            input_array = np.array(input_img)

            H, W = input_array.shape[:2]
            input_grey = gray(input_array)
            input_bin = binarization(input_grey, 5, 0.15)
            input_bin[input_bin == 0] = 1
            input_bin[input_bin == 255] = 0

            weight = get_weight(input_bin)
            relative_weight = weight / (W * H)

            weight_1 = get_weight(input_bin, 0, W // 2, 0, H // 2)
            relative_weight_1 = weight_1 / ((W // 2) * (H // 2))

            weight_2 = get_weight(input_bin, W // 2, W, 0, H // 2)
            relative_weight_2 = weight_2 / ((W - W // 2) * (H // 2))

            weight_3 = get_weight(input_bin, 0, W // 2, H // 2, H)
            relative_weight_3 = weight_3 / ((W // 2) * (H - H // 2))

            weight_4 = get_weight(input_bin, W // 2, W, H // 2, H)
            relative_weight_4 = weight_4 / ((W - W // 2) * (H - H // 2))

            center_x, center_y = get_center(input_bin, weight)
            relative_center_x = (center_x - 1) / (W - 1)
            relative_center_y = (center_y - 1) / (H - 1)

            inertia_x, inertia_y = get_inertia(input_bin, center_x, center_y)
            relative_inertia_x = inertia_x / (W**2 * H**2)
            relative_inertia_y = inertia_y / (W**2 * H**2)

            data = [
                alphabet[i-1], weight, relative_weight, weight_1, relative_weight_1, weight_2, relative_weight_2,
                weight_3, relative_weight_3, weight_4, relative_weight_4, center_x, center_y,
                relative_center_x, relative_center_y,
                inertia_x, inertia_y, relative_inertia_x, relative_inertia_y
            ]
            writer.writerow(data)
            write_profile_x(input_bin, i)
            write_profile_y(input_bin, i)

            outputStr = os.path.join(output_folder, file_name)
            invert_img = invert(input_img)
            invert_img.save(outputStr)