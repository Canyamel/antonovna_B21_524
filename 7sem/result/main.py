import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from alphabet import GEORGIAN
import pandas as pd
import os
from binarization import gray, binarization
from characteristic import get_features_letter

def get_distance(vector1, vector2) -> int:
    assert len(vector1) == len(vector2), "Длина векторов должна быть одинаковым"

    dist = 0
    for i in range(len(vector1)):
        dist += (vector1[i] - vector2[i]) ** 2

    return dist**(1/2)

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(current_dir, 'input/56')
    output_path = os.path.join(current_dir, 'output/56')

    original_phrase = "შეუძლებელიაგიყვარდესდაიყობრძენი"
    table = pd.read_csv(os.path.join(input_path, 'table.csv'))
    feature_names = ['relative_' + name for name in ['weight_1','weight_2','weight_3','weight_4', 'center_x', 'center_y', 'inertia_x', 'inertia_y']]

    len_original_phrase = len(original_phrase)

    path = os.path.join(output_path, 'proximity.txt')
    if os.path.exists(path):
        os.remove(path)

    for i in range(len_original_phrase):
        letter_img = Image.open(os.path.join(input_path, 'letters/letter_' + str(i+1) + '.png')).convert('RGB')
        letter_array = np.array(letter_img)
        letter_gray = gray(letter_array)
        letter_bin = binarization(letter_gray, 5, 0.15)
        letter_bin[letter_bin == 0] = 1
        letter_bin[letter_bin == 255] = 0
        features_letter = get_features_letter(letter_bin)

        distances = table.apply(
            lambda feature_symbol: get_distance(feature_symbol[feature_names], features_letter), axis = 1
        )

        proximities = distances.apply(
            lambda distance: 1 - distance / distances.max()
        )
        proximities = pd.concat([table.letter, proximities], axis=1)
        proximities.columns = ['letter', 'proximity']
        proximities = proximities.sort_values('proximity', ascending=False)
        proximities = proximities.reset_index(drop=True)

        data = []
        for j in range(len(proximities)):
            data.append((proximities['letter'][j], proximities['proximity'][j]))

        with open(os.path.join(output_path, "proximity.txt"), "a", encoding="utf-8") as file:
            file.write(", ".join(str(j) for j in data))
            file.write("\n")

        result_phrase = ""
        with open(os.path.join(output_path, "proximity.txt"), "r", encoding="utf-8") as file:
            for line in file:
                start = line.find("(")
                end = line.find(")")

                value = line[start + 1 : end]
                result_phrase += value[1]

    mistake = 0
    for i in range(0, len_original_phrase):
        if original_phrase[i] != result_phrase[i]:
            mistake += 1
            print("====")
            print("Порядковый номер буквы:", i+1)
            print("Должно быть:", original_phrase[i])
            print("Получили:", result_phrase[i])
            print("====")
    if mistake ==  0:
        print("Ошибок нет")
    print("Верно", len_original_phrase - mistake, 'из',  len_original_phrase)
    percent = ((len_original_phrase - mistake) / len_original_phrase) * 100
    print("Доля верно верно распознанных символов:", percent, "%")

if __name__ == "__main__":
    main()