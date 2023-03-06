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


if __name__ == '__main__':
    root_dir = "C:/Users/weber/PycharmProjects/playground/OCR_data/"

    df = read_dataset("C:/Users/weber/PycharmProjects/playground/catalog.csv")
    processor = initial_processor("microsoft/trocr-small-printed")

    map_dataset = build_map_dataset(root_dir, df, processor)
    iter_dataset = build_iter_dataset(root_dir, df, processor)


