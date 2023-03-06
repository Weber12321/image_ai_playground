import json

import cv2

import numpy as np
import pandas as pd

from PIL import ImageFont, ImageDraw, Image


def catalog_process(path, font_size, font_family, save_name):
    df = pd.read_csv(path, encoding='utf-8')
    sub_df = df.loc[(df['font_family'] == font_family) & (df['font_size'] == font_size)]
    sub_df.to_csv(save_name, encoding='utf-8-sig', index=False)


def read_img(path):
    if path.endswith('.json'):
        with open(path, 'r') as f:
            a = np.array(json.load(f))
            a = a.astype(np.uint8)
            b = Image.fromarray(a)
    else:
        a = cv2.imread(path, cv2.IMREAD_COLOR)
        b = Image.open(path).convert("RGB")

    print((np.asarray(b)[:, :, [2, 1, 0]] == a).all())

    b = b.resize((768, 24))
    a = cv2.resize(a, (768, 24))

    b = np.asarray(b)[:, :, [2, 1, 0]]
    print((b - a).mean())

    # cv2.imshow('My Image', a)
    cv2.imshow('My Image', np.array(b))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def draw(p):
    font_path = 'OCR_Training_Font/標楷體/kaiu.ttf'
    font = ImageFont.truetype(font_path, 12)

    img = np.zeros((12, 384, 3), np.uint8)
    img[:] = (255, 255, 255)
    # df = pd.read_csv('教育部4808個常用字.csv', encoding='utf-8')

    imgPil = Image.fromarray(img)
    draw = ImageDraw.Draw(imgPil)
    draw.text((-1, -1), p, font=font, fill=(0, 0, 0))

    img = np.array(imgPil)
    # with open('expected_arr.json', 'w') as f:
    #     json.dump(img.tolist(), f)
    M = np.float32([[1, 0, 100], [0, 1, 0], [0, 0, 1]])

    cv2.imshow('My Image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # # cv2.imwrite('output2.jpg', img)

if __name__ == '__main__':
    # p = '龜'
    # draw(p)
    # catalog_process('catalog.csv', 'kaiu', 36, 'catalog_small.csv')
    read_img('expected_arr.json')
