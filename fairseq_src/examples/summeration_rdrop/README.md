# Fine-tuning BART on CNN-Dailymail summarization task

# Prepare Data
Follow the instructions [here](https://github.com/abisee/cnn-dailymail) to download the original CNN and Daily Mail datasets. To preprocess the data, refer to the pointers in [this issue](https://github.com/pytorch/fairseq/issues/1391) or check out the code [here](https://github.com/artmatsak/cnn-dailymail).

Follow the instructions [here](https://github.com/EdinburghNLP/XSum) to download the original Extreme Summarization datasets, or check out the code [here](https://github.com/EdinburghNLP/XSum/tree/master/XSum-Dataset), Please keep the raw dataset and make sure no tokenization nor BPE on the dataset.

# BPE preprocess and Binarize the data:

```
bash script/preprocess.sh
```

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
files2rouge /data/lxb/cnndailymail_sum/cnn_dm/test.target /data/lxb/cnndailymail_sum/cnn_dm/test_small_kl.hypo
```
