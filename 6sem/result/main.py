import numpy as np
from PIL import Image
import os
from help import gray, binarization, get_borders_symbol, get_borders_line, get_borders_text, write_profile_x, write_profile_y

def main():
    current_folder = os.path.dirname(os.path.abspath(__file__))
    input_folder = os.path.join(current_folder, 'input')
    output_folder = os.path.join(current_folder, 'output/letter')

    file_name = 'phrase.png'
    input_str = os.path.join(input_folder, file_name)
    input_img = Image.open(input_str).convert('RGB')
    input_array = np.array(input_img)

    grey_array = gray(input_array)
    input_bin = binarization(grey_array, 5, 0.15)

    input_bin[input_bin == 0] = 1
    input_bin[input_bin == 255] = 0

    borders_symbol = get_borders_symbol(input_bin)
    for i in range(0, len(borders_symbol)):
        letter_array = input_array[borders_symbol[i][0][1] : borders_symbol[i][1][1] + 1, borders_symbol[i][0][0] : borders_symbol[i][1][0] + 1]
        letter_img = Image.fromarray(letter_array)
        output_path = os.path.join(output_folder, f'letter_{i+1}.png')
        letter_img.save(output_path)

    output_folder = os.path.join(current_folder, 'output')

    file_name = 'phrase_2.png'
    input_str = os.path.join(input_folder, file_name)
    input_img = Image.open(input_str).convert('RGB')
    input_array = np.array(input_img)

    grey_array = gray(input_array)
    input_bin = binarization(grey_array, 5, 0.15)

    input_bin[input_bin == 0] = 1
    input_bin[input_bin == 255] = 0

    borders_text = get_borders_text(input_bin)
    text_array = input_array[borders_text[0][1] : borders_text[1][1] + 1, borders_text[0][0] : borders_text[1][0] + 1]
    text_img = Image.fromarray(text_array)
    output_path = os.path.join(output_folder, f'text.png')
    text_img.save(output_path)

    borders_line = get_borders_line(input_bin)
    for i in range(0, len(borders_line)):
        line_array = input_array[borders_line[i][0][1] : borders_line[i][1][1] + 1, borders_line[i][0][0] : borders_line[i][1][0] + 1]
        line_img = Image.fromarray(line_array)
        output_path = os.path.join(output_folder, f'line_{i+1}.png')
        line_img.save(output_path)

if __name__ == '__main__':
    main()