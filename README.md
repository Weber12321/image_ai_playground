# image_ai_playground

###### env: Python 3.8, windows 11

## Usage
+ Setup the environment: 
```bash
$ git clone https://github.com/Weber12321/image_ai_playground.git 

$ cd image_ai_playground

$ virtualenv venv
# or `python -m venv venv`
$ pip install -r requirements.base.txt
```
+ configure:
  + Make sure you have downloaded the requirements Chinese datasets and catalog.
    + Please contact to the author for accessing the Chinese dataset.
  + Create a folder named, model_output, in the project directory.
    + Create a folder named, state, under the model_output.
    + Create a folder named, torch_script, under the model_output.
  + modify the path inside `config.py` in the project root path
    + `OCR_DATA_PATH` set to your ocr dataset base path.
    + `DUMMY_DATA_PATH` set to your ocr small dataset base path.
    + `OCR_CATALOG` set to your catalog file path, used for training.
  
+ Run the training script:
  + The script is wrapped with python CLI tool, click, please modify the options if you want to change the training arguments.
```bash
$ python run_trocr.py 
```