import os
from datetime import datetime

import click
import numpy as np
import pandas as pd
import torch
import wandb
from torch.utils.data import DataLoader, SubsetRandomSampler

from transformers import AdamW, get_scheduler
from transformers import logging as hf_logging
from tqdm import tqdm

from config import MODEL_BIN_PATH, MODEL_STATE_PATH, OCR_CATALOG, \
    OCR_DATA_PATH, VALI_CATALOG, VALI_DATA_PATH, PREP_DIR
from trocr.data import ZhPrintedDataset
from trocr.metric import prf_metric
from trocr.model import initial_model, initial_processor

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


@click.command()
@click.option('--task_name', default='NaN')
@click.option('--trace/--no-trace', default=True)
@click.option('--max_target_length', default=20)
@click.option('--device', default="cuda")
@click.option('--model_ckpt', default="microsoft/trocr-small-printed")
@click.option('--epochs', default=1000)
@click.option('--lr', default=0.000001)
@click.option('--batch_size', default=16)
@click.option('--model_state_dir', default=None)
@click.option('--num_training_steps_per_epoch', default=20)
@click.option('--num_validation_steps_per_epoch', default=10)
@click.option('--load_state/--no_load_state', default=True)
@click.option('--save_bin_with_loss/--save_bin_with_score', default=True)
def run(
    task_name, trace, max_target_length, device,
    model_ckpt, epochs, lr, batch_size, model_state_dir,
    num_training_steps_per_epoch, num_validation_steps_per_epoch, load_state,
    save_bin_with_loss
):
    hf_logging.set_verbosity_error()

    catalog_path = OCR_CATALOG
    vali_path = VALI_CATALOG
    data_path = OCR_DATA_PATH
    vali_data_path = VALI_DATA_PATH
    if not model_state_dir:
        model_state_dir = MODEL_STATE_PATH
    if task_name == 'NaN':
        task_ = datetime.now().strftime('%Y-%m-%d-%H-%M')
    else:
        task_ = task_name

    if trace:
        wandb.init(project="trocr_tiny_expr", entity="weber12321", name=task_)

    if load_state:
        bin_path = MODEL_BIN_PATH

    click.echo('Load data catalog and dataset...')
    train_df = pd.read_csv(catalog_path, encoding='utf-8')
    test_df = pd.read_csv(vali_path, encoding='utf-8')

    click.echo('Init processor...')
    processor = initial_processor(model_ckpt=PREP_DIR, local=True)

    click.echo('Setup training and testing dataset...')
    train_dataset = ZhPrintedDataset(
        root_dir=data_path,
        df=train_df,
        processor=processor,
        max_target_length=int(max_target_length)
    )

    test_dataset = ZhPrintedDataset(
        root_dir=vali_data_path,
        df=test_df,
        processor=processor,
        max_target_length=int(max_target_length)
    )


    click.echo(f"Number of training examples: {len(train_dataset)}")
    # click.echo(f"Number of validation examples: {len(test_dataset)}")

    # torch.manual_seed(42)

    # set subset sampler
    # train_subset_sampler = SubsetRandomSampler(
    #     np.random.choice(len(train_dataset), 16000, replace=False)
    # )
    # test_subset_sampler = SubsetRandomSampler(
    #     np.random.choice(len(test_dataset), 1000, replace=False)
    # )
    click.echo(f"Number of validation examples: {len(test_dataset)}")
    # train_subset = Subset(train_dataset, )

    train_dataloader = DataLoader(train_dataset, batch_size=int(batch_size), num_workers=6,
                                  shuffle=True)
    # train_dataloader = DataLoader(train_dataset, batch_size=int(batch_size), sampler=train_subset_sampler)
    test_dataloader = DataLoader(test_dataset, batch_size=int(batch_size), num_workers=6,
                                 shuffle=True)

    click.echo('Initializing model...')
    model = initial_model(model_ckpt=model_ckpt)
    model.to(device)

    # update the embedding size due to new custom tokenizer vocab
    model.decoder.resize_token_embeddings(len(processor.tokenizer))

    # set special tokens used for creating the decoder_input_ids from the labels
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    # make sure vocab size is set correctly
    model.config.vocab_size = model.config.decoder.vocab_size

    # set beam search parameters
    model.config.eos_token_id = processor.tokenizer.sep_token_id
    model.config.max_length = 15
    model.config.early_stopping = True
    model.config.no_repeat_ngram_size = 3
    model.config.length_penalty = 2.0
    model.config.num_beams = 4

    if load_state:
        click.echo(f'Loading the model weight from {bin_path} ...')
        model.load_state_dict(torch.load(bin_path))

    # traced_model = torch.jit.trace(model, dummy_input)
    # save_model_path_pt = os.path.join(model_save_dir, f"{task_}.bin")
    # torch.jit.save(traced_model, save_model_path_pt)

    if trace:
        wandb.config.update({
            "task": task_,
            "length of training_data": len(train_dataset),
            "length of test_data": len(test_dataset),
            "learning_rate": float(lr),
            "epochs": epochs,
            "batch_size": batch_size,
            "max_target_length": int(max_target_length),
            "pre-trained model": model_ckpt,
            "device": device
        })
        wandb.config.update({
            "model_details": model.config.to_dict()
        })

    optimizer = AdamW(model.parameters(), lr=float(lr))
    # num_training_steps = num_training_steps
    click.echo(f"num_training_steps_per_epoch: {num_training_steps_per_epoch}")
    click.echo(
        f"num_validation_steps_per_epoch: {num_validation_steps_per_epoch}")
    # num_training_steps = int(num_training_steps)
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=int(len(train_dataloader))
    )

    def next_(loader):
        while True:
            for batch in loader:
                yield batch

    click.echo('Start training...')

    save_model_path = os.path.join(model_state_dir, f"{task_}")
    os.makedirs(save_model_path, exist_ok=True)

    train_iterator_ = next_(train_dataloader)
    test_iterator_ = next_(test_dataloader)
    for epoch in range(1, int(epochs) + 1):
        # train
        click.echo(f" ==== epoch {epoch} ==== ")
        model.train()
        train_loss = 0.0

        for _ in tqdm(range(num_training_steps_per_epoch)):
            batch = next(train_iterator_)
            for k, v in batch.items():
                batch[k] = v.to(device)

            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            train_loss += loss.item()

        click.echo(
            f" ==== Loss of epoch {epoch}: {train_loss / num_training_steps_per_epoch} ==== "
        )

        # torch.save(model.state_dict(), save_model_path)
        # evaluate
        precision, recall, f1 = 0.0, 0.0, 0.0
        model.eval()
        vali_loss = 0.0
        temp_loss = 0.0
        with torch.no_grad():
            for idx in tqdm(range(num_validation_steps_per_epoch)):
                batch = next(test_iterator_)
                # run batch generation
                for k, v in batch.items():
                    batch[k] = v.to(device)
                outputs = model(**batch)
                loss = outputs.loss
                vali_loss += loss.item()

                outputs = model.generate(batch["pixel_values"].to(device))

                p, r, f = prf_metric(pred_ids=outputs,
                                     label_ids=batch["labels"],
                                     processor=processor)
                if f > f1:
                    precision = p
                    recall = r
                    
                    if not save_bin_with_loss:
                        torch.save(
                            model.state_dict(),
                            save_model_path + f"_f1_{round(f, 4)}.bin"
                        )
                        os.remove(
                            save_model_path + f"_f1_{round(f1, 4)}.bin"
                        )
                    f1 = f

                if save_bin_with_loss:
                    if idx == 0:
                        temp_loss = loss.item()

                    if loss.item() <= temp_loss:
                        
                        torch.save(
                            model.state_dict(),
                            save_model_path + f"_val_loss_{round(loss.item(), 4)}.bin"
                        )
                        if idx != 0:
                            os.remove(
                                save_model_path + f"_val_loss_{round(temp_loss, 4)}.bin"  
                            )

                        temp_loss = loss.item()

        click.echo(
            f" ==== Loss of epoch {epoch}: {vali_loss / num_validation_steps_per_epoch} ==== ")
        click.echo(
            f" ==== Validation score: precision {precision}, recall {recall}, f1 {f1} ==== ")

        if trace:
            wandb.log({
                "epoch": epoch,
                "train_loss": train_loss / num_training_steps_per_epoch,
                "vali_loss": vali_loss / num_validation_steps_per_epoch,
                "precision": precision,
                "recall": recall,
                "f1": f1
            })


if __name__ == '__main__':
    run()
