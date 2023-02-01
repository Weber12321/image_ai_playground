import json

import cv2

import numpy as np
import pandas as pd

from PIL import ImageFont, ImageDraw, Image



def draw(p):
    font_path = 'OCR_Training_Font/標楷體/kaiu.ttf'
    font = ImageFont.truetype(font_path, 36)

    img = np.zeros((38, 384, 3), np.uint8)
    img[:] = (255, 255, 255)
    # df = pd.read_csv('教育部4808個常用字.csv', encoding='utf-8')

    imgPil = Image.fromarray(img)
    draw = ImageDraw.Draw(imgPil)
    draw.text((0, 0), p, font=font, fill=(0, 0, 0))

    img = np.array(imgPil)
    # with open('expected_arr.json', 'w') as f:
    #     json.dump(img.tolist(), f)
    cv2.imshow('My Image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # cv2.imwrite('output2.jpg', img)

if __name__ == '__main__':
    p = '龜'
    s = draw(p)

