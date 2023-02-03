import os
from datetime import datetime

import click
import torch
import wandb
from torch.utils.data import DataLoader

from transformers import AdamW, get_scheduler
from tqdm import tqdm

from config import DUMMY_DATA_PATH, MODEL_SAVE_PATH, MODEL_STATE_PATH, OCR_CATALOG, OCR_DATA_PATH
from trocr.data import ZhPrintedDataset
from trocr.metric import prf_metric
from trocr.model import initial_model, initial_processor
from trocr.process import read_dataset, train_test_split_dataset
from trocr.save import save_pt


@click.command()
@click.option('--task_name', default='NaN')
@click.option('--trace/--no-trace', default=False)
@click.option('--catalog_path', default=None)
@click.option('--data_path', default=None)
@click.option('--max_target_length', default=20)
@click.option('--device', default="cpu")
@click.option('--model_ckpt', default="microsoft/trocr-small-printed")
@click.option('--epochs', default=3)
@click.option('--lr', default=0.00005)
@click.option('--batch_size', default=32)
@click.option('--model_state_dir', default=None)
@click.option('--model_save_dir', default=None)
def run(
    task_name, trace, catalog_path, data_path, max_target_length, device,
    model_ckpt, epochs, lr, batch_size, model_state_dir, model_save_dir
):
    if not catalog_path:
        catalog_path = OCR_CATALOG
    if not data_path:
        # data_path = OCR_DATA_PATH
        data_path = DUMMY_DATA_PATH
    if not model_state_dir:
        model_state_dir = MODEL_STATE_PATH
    if not model_save_dir:
        model_save_dir = MODEL_SAVE_PATH

    if task_name == 'NaN':
        task_ = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    else:
        task_ = task_name

    if trace:
        wandb.init(project="trocr", entity="weber12321")

    click.echo('Load data catalog and split dataset...')
    df = read_dataset(catalog_path)
    train_df, test_df = train_test_split_dataset(df, test_size=0.1)

    click.echo('Init processor...')
    processor = initial_processor(model_ckpt=model_ckpt)

    click.echo('Setup training and testing dataset...')
    train_dataset = ZhPrintedDataset(
        root_dir=data_path,
        df=train_df,
        processor=processor,
        max_target_length=int(max_target_length)
    )

    test_dataset = ZhPrintedDataset(
        root_dir=data_path,
        df=test_df,
        processor=processor,
        max_target_length = int(max_target_length)
    )

    dummy_input = train_dataset[0]['pixel_values']

    click.echo(f"Number of training examples: {len(train_dataset)}")
    click.echo(f"Number of validation examples: {len(test_dataset)}")

    train_dataloader = DataLoader(train_dataset, batch_size=int(batch_size), shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=int(batch_size))

    click.echo('Initializing model...')
    model = initial_model(model_ckpt=model_ckpt)
    model.to(device)

    # set special tokens used for creating the decoder_input_ids from the labels
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    # make sure vocab size is set correctly
    model.config.vocab_size = model.config.decoder.vocab_size

    # set beam search parameters
    model.config.eos_token_id = processor.tokenizer.sep_token_id
    model.config.max_length = 64
    model.config.early_stopping = True
    model.config.no_repeat_ngram_size = 3
    model.config.length_penalty = 2.0
    model.config.num_beams = 4

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
    num_training_steps = int(epochs) * len(train_dataloader)
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )
    click.echo('Start training...')

    save_model_path = os.path.join(model_state_dir, f"{task_}.bin")
    precision, recall, f1 = 0.0, 0.0, 0.0
    for epoch in range(int(epochs)):
        # train
        click.echo(f" ==== epoch {epoch} ==== ")
        model.train()
        train_loss = 0.0
        for batch in tqdm(train_dataloader):
            # get the inputs
            for k, v in batch.items():
                batch[k] = v.to(device)

            # forward + backward + optimize
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            train_loss += loss.item()

        click.echo(f" ==== Loss after epoch {epoch}: {train_loss / len(train_dataloader)} ==== ", )

        # evaluate
        model.eval()

        with torch.no_grad():
            for batch in tqdm(test_dataloader):
                # run batch generation
                outputs = model.generate(batch["pixel_values"].to(device))
                # compute metrics
                p, r, f = prf_metric(pred_ids=outputs, label_ids=batch["labels"], processor=processor)
                if f > f1:
                    torch.save(model.state_dict(), save_model_path)
                    wandb.save(f"{task_}.bin")

        precision /= len(test_dataloader)
        recall /= len(test_dataloader)
        f1 /= len(test_dataloader)

        click.echo(f" ==== Validation score: precision {precision}, recall {recall}, f1 {f1} ==== ")

        if trace:
            wandb.log({
                "epoch": epoch,
                "train_loss": train_loss / len(train_dataloader),
                "precision": precision,
                "recall": recall,
                "f1": f1
            })

    save_model_path_pt = os.path.join(model_save_dir, f"{task_}.bin")
    save_pt(model, dummy_input, save_model_path, save_model_path_pt)


if __name__ == '__main__':
    run()