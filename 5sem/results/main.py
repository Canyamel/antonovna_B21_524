from generate import generate_table
import csv
from alphabet import GEORGIAN, FONT_NAME, FONT_SIZE, LANGUAGE

def main():
    generate_table(GEORGIAN)

    with open('./lab5/output/table.csv', 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        with open('./lab5/README.md', 'a+', encoding='utf-8') as file:
            next(reader)
            file.write(
                f"""### Язык {LANGUAGE}
### Название шрифта {FONT_NAME}
### Размер шрифта {FONT_SIZE}
"""
            )
            for (i, letter), row in zip(enumerate(GEORGIAN), reader):
                file.write(
                    f"""
### Буква {letter}

<img src="./input/{i+1}.png" width="150"> <img src="./output/invert/{i+1}.png" width="150"> <img src="output/x/{i+1}.png" width="300"> <img src="output/y/{i+1}.png" width="300">
"""
                )

                file.write(
                    f"""
Признаки:
1. Вес чёрного = {row[1]}
2. Нормированный вес чёрного = {row[2]}
3. Вес чёрного (I четверть) = {row[3]}
4. Нормированный вес чёрного (I четверть) = {row[4]}
5. Вес чёрного (II четверть) = {row[5]}
6. Нормированный вес чёрного (II четверть) = {row[6]}
7. Вес чёрного (III четверть) = {row[7]}
8. Нормированный вес чёрного (III четверть) = {row[8]}
9. Вес чёрного (IV четверть) = {row[9]}
10. Нормированный вес чёрного (IV четверть) = {row[10]}
11. Центр масс = ({row[11]}, {row[12]})
12. Нормированный центр масс = ({row[13]}, {row[14]})
13. Моменты инерции = ({row[15]}, {row[16]})
14. Нормированные моменты инерции = ({row[17]}, {row[18]})
"""
                )

if __name__ == '__main__':
    main()