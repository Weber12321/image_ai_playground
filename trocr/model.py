import torch
from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel,
)

def initial_model(model_ckpt=None):
    if model_ckpt:
        model = VisionEncoderDecoderModel.from_pretrained(model_ckpt)
    else:
        raise ValueError("model_ckpt is unknown.")

    return model

def initial_processor(model_ckpt=None, local=True):
    if model_ckpt:
        if local:
            processor = TrOCRProcessor.from_pretrained(model_ckpt, local_files_only=True)
        else:
            processor = TrOCRProcessor.from_pretrained(model_ckpt)
    else:
        raise ValueError("model_ckpt is unknown.")

    return processor
