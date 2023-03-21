from enum import Enum
from typing import Optional, Union
import torch
import numpy as np
from torch.utils.data import Dataset, IterableDataset
from PIL import Image


class ChannelDimension(Enum):
    FIRST = "channels_first"
    LAST = "channels_last"

class ZhPrintedDataset(Dataset):
    def __init__(self, root_dir, df, processor, max_target_length=10):
        self.root_dir = root_dir
        self.df = df
        self.processor = processor
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.df)

    def crop_whitespaces(self, img):
        arr = self.crop_image(np.array(img))
        return self.to_pil_image(arr)
    
    def crop_image(self, img):
        mask = img!=255
        mask = mask.any(2)
        mask0,mask1 = mask.any(0),mask.any(1)
        return img[np.ix_(mask1,mask0)]
    
    def rescale(
        self,
        image: np.ndarray,
        scale: Union[int, float],
        data_format: Optional[ChannelDimension] = None,
        dtype=np.float32
    ) -> np.ndarray:

        rescaled_image = image * scale
        if data_format is not None:
            rescaled_image = self.to_channel_dimension_format(
                rescaled_image, data_format
            )
        rescaled_image = rescaled_image.astype(dtype)
        return rescaled_image

    def to_channel_dimension_format(
        self,
        image: np.ndarray,
        channel_dim: Union[ChannelDimension, str],
        input_channel_dim: Optional[Union[ChannelDimension, str]] = None,
    ) -> np.ndarray:
        """
        Converts `image` to the channel dimension format  
        specified by `channel_dim`.
        Args:
            image (`numpy.ndarray`):
                The image to have its channel dimension set.
            channel_dim (`ChannelDimension`):
                The channel dimension format to use.
        Returns:
            `np.ndarray`: The image with the channel dimension 
            set to `channel_dim`.
        """
        if not isinstance(image, np.ndarray):
            raise ValueError(
                f"Input image must be of type np.ndarray, got {type(image)}"
            )

        if input_channel_dim is None:
            input_channel_dim = self.infer_channel_dimension_format(image)

        target_channel_dim = ChannelDimension(channel_dim)
        if input_channel_dim == target_channel_dim:
            return image

        if target_channel_dim == ChannelDimension.FIRST:
            image = image.transpose((2, 0, 1))
        elif target_channel_dim == ChannelDimension.LAST:
            image = image.transpose((1, 2, 0))
        else:
            raise ValueError(
                f"Unsupported channel dimension format: {channel_dim}"
            )

        return image

    def to_pil_image(self, img):
        do_rescale = img.flat[0].dtype != np.uint8

        if do_rescale:
            data = self.rescale(data, 255)
        data = data.astype(np.uint8)
        return Image.fromarray(data)

    def __getitem__(self, idx):
        # get file name + text
        file_name = self.df['image_path'][idx]
        text_file_name = file_name.split('.')[0] + '.txt'
        # text = self.df['text'][idx]
        with open(self.root_dir + text_file_name, 'r', encoding='utf-8') as f:
            text = f.read()
        # prepare image (i.e. resize + normalize)
        image = Image.open(self.root_dir + file_name).convert("RGB")
        # crop out whitespaces
        image = self.crop_whitespaces(image)
        pixel_values = self.processor(image, return_tensors="pt").pixel_values
        # add labels (input_ids) by encoding the text
        labels = self.processor.tokenizer(text,
                                          padding="max_length",
                                          max_length=self.max_target_length).input_ids
        # important: make sure that PAD tokens are ignored by the loss function
        labels = [label if label != self.processor.tokenizer.pad_token_id else -100 for label in labels]

        encoding = {"pixel_values": pixel_values.squeeze(), "labels": torch.tensor(labels)}
        return encoding


class ZhPrintedIterDataset(IterableDataset):
    def __init__(self, root_dir, df, processor, max_target_length=10):
        self.root_dir = root_dir
        self.df = df
        self.processor = processor
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.df)

    def __iter__(self):
        # get file name + text
        for _, row in self.df.iterrows():

            file_name = row['image_path']
            text = row['text']
            # prepare image (i.e. resize + normalize)
            image = Image.open(self.root_dir + file_name).convert("RGB")
            pixel_values = self.processor(image, return_tensors="pt").pixel_values
            # add labels (input_ids) by encoding the text
            labels = self.processor.tokenizer(text,
                                              padding="max_length",
                                              max_length=self.max_target_length).input_ids
            # important: make sure that PAD tokens are ignored by the loss function
            labels = [label if label != self.processor.tokenizer.pad_token_id else -100 for label in labels]

            encoding = {"pixel_values": pixel_values.squeeze(), "labels": torch.tensor(labels)}
            yield encoding
