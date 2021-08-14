# Fine-tuning BART on CNN-Dailymail summarization task

# Prepare Data
Follow the instructions [here](https://github.com/abisee/cnn-dailymail) to download the original CNN and Daily Mail datasets. To preprocess the data, refer to the pointers in [this issue](https://github.com/pytorch/fairseq/issues/1391) or check out the code [here](https://github.com/artmatsak/cnn-dailymail).

# BPE preprocess and Binarize the data:

```
bash script/preprocess.sh
```

## Pre-trained models

Model | Description | # params | Download
---|---|---|---
`bart.base` | BART model with 6 encoder and decoder layers | 140M | [bart.base.tar.gz](https://dl.fbaipublicfiles.com/fairseq/models/bart.base.tar.gz)
`bart.large` | BART model with 12 encoder and decoder layers | 400M | [bart.large.tar.gz](https://dl.fbaipublicfiles.com/fairseq/models/bart.large.tar.gz)

# Training Script

Example fine-tuning CNN-DM
```
bash script/run_train.sh
```

# Inference Script

After training the model as mentioned in previous step, you can perform inference with checkpoints in checkpoints/ directory using run_inference.py, for example:

```
python script/run_inference.py
```

Then, Download [files2rouge](https://github.com/pltrdy/files2rouge) to evaluate the result by ROUGLE-x:
```
files2rouge /data/cnn_dm/test.target /data/cnn_dm/test.hypo
```

> Note that our environment: GPU GeForce RTX 3090 (24G) NVIDIA Driver Version = 460.67  CUDA Version = 11.2 torch version = 1.8.1 .
> We are not sure whether it will work for other environments. If you don't have enough GPU memory, you can modify parameter: MAX_TOKENS=1024
UPDATE_FREQ=16 to ease this problem.