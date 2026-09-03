#!/usr/bin/env python3
"""Generate the downloadable Excel example used by the regression page."""

from pathlib import Path

from openpyxl import Workbook


OUTPUT_PATH = (
    Path(__file__).resolve().parents[1]
    / 'frontend'
    / 'static'
    / 'examples'
    / 'linear_regression_example.xlsx'
)


def main():
    workbook = Workbook()
    worksheet = workbook.active
    worksheet.title = '房价示例'
    worksheet.append(['房屋面积', '房龄', '距市中心距离', '房价'])

    noise = [2.0, -1.5, 3.2, -2.3, 0.8, 1.6, -0.9, 2.7, -1.8, 0.4]
    for index in range(30):
        area = 62 + index * 4 + (index % 3) * 3
        age = 2 + (index * 7) % 28
        distance = 1.2 + (index * 1.7) % 15
        price = 20 + 0.85 * area - 0.6 * age - 2.5 * distance + noise[index % len(noise)]
        worksheet.append([area, age, round(distance, 1), round(price, 2)])

    worksheet.freeze_panes = 'A2'
    for column, width in {'A': 14, 'B': 10, 'C': 18, 'D': 12}.items():
        worksheet.column_dimensions[column].width = width

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    workbook.save(OUTPUT_PATH)
    print(OUTPUT_PATH)


if __name__ == '__main__':
    main()
