import torch


def save_pt(model, dummy_input, state_dict_path, pt_output_path):
    # https://huggingface.co/docs/transformers/serialization#using-torchscript-in-python
    # https://pytorch.org/tutorials/beginner/saving_loading_models.html#export-load-model-in-torchscript-format
    # https://huggingface.co/docs/transformers/serialization#saving-a-model
    model.load_state_dict(torch.load(state_dict_path))
    traced_model = torch.jit.trace(model, dummy_input)
    torch.jit.save(traced_model, pt_output_path)