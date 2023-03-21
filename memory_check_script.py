import cv2
import numpy as np

from trocr.data import *
from trocr.model import initial_processor
from trocr.process import read_dataset
from memory_profiler import profile

@profile
def build_map_dataset(root_dir, catalog, processor):
    return ZhPrintedDataset(root_dir, catalog, processor)

@profile
def build_iter_dataset(root_dir, catalog, processor):
    return ZhPrintedIterDataset(root_dir, catalog, processor)


def crop_whitespace(image_path):
    def crop_image(img):
        mask = img != 255
        mask = mask.any(2)
        mask0, mask1 = mask.any(0), mask.any(1)
        return img[np.ix_(mask1, mask0)]

    img = Image.open(image_path).convert("RGB")


    cropped = crop_image(np.array(img))

    print(cropped.shape)
    Image.fromarray(cropped)
    # cv2.imwrite('temp.png', cropped)



if __name__ == '__main__':
    # root_dir = "/root/algo_rd/image_ai_playground/vali_data/"
    #
    # df = read_dataset("/root/algo_rd/image_ai_playground/vali_catalog.csv")
    # processor = initial_processor("microsoft/trocr-small-printed")
    #
    # map_dataset = ZhPrintedDataset(root_dir=root_dir, df=df, processor=processor, max_target_length=20)
    # print(map_dataset[0])

    crop_whitespace('OCR_data/mingliu/12/0ad30846-d8c8-44f6-a64e-b08e6bc1832a.png')
