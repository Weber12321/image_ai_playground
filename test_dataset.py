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
    root_dir = "/root/algo_rd/image_ai_playground/vali_data/"

    df = read_dataset("/root/algo_rd/image_ai_playground/vali_catalog.csv")
    processor = initial_processor("microsoft/trocr-small-printed")

    map_dataset = ZhPrintedDataset(root_dir=root_dir, df=df, processor=processor, max_target_length=20)
    print(map_dataset[0])


