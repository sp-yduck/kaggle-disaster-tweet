# Natural Language Processing with Disaster Tweets (Kaggle)

- Competition site URL (https://www.kaggle.com/c/nlp-getting-started)

## How to Use

### installatino requirements

- step0 (Optional): set up virtual environment

  ```bash
  python -m venv [A directory to create the environment in]
  ```

  Windows

  ```bash
  source [same as the upper one]/Scripts/activate
  ```

  Mac/Linux

  ```bash
  source [same as the upper one]/bin/activate
  ```

- step1: install python packages

  ```bash
  pip install -r requirements.txt
  ```

- step2: install pytorch and cuda

  see [pytorch.org](https://pytorch.org/get-started/locally/)

  If you use Windows and CUDA:v11

  ```bash
  pip3 install torch==1.10.2+cu113 torchvision==0.11.3+cu113 torchaudio===0.10.2+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
  ```

### preprocess (data splitting)

data/nlp-getting-started/train.csv -> data/inputs/[train, val, test].csv

```bash
python data/preprocess.py
```

### training

```bash
python main.py [config file path] --train
```

### testing

```bash
python main.py [config file path] --test
```


# License
The source code is licensed MIT.
see LICENSE.
