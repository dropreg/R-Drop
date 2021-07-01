
# Neural Machine Translation

This example contains instructions for training a new NMT model with R-Drop.

# Prepare Data
The following instructions can be used to train a Transformer model on the [IWSLT'14 German to English dataset](http://workshop2014.iwslt.org/downloads/proceeding.pdf).

Download and preprocess the data:
```bash
bash script/prepare-iwslt14.sh

```

Binarize the dataset
```
cd ../../
bash examples/translation_rdrop/script/run_binarize.sh
```

# Training Script

Train a Transformer translation model with R-Drop over this data:
```
bash examples/translation_rdrop/script/run_train.sh
```

# Inference Script

Evaluate R-Drop model over this data:
```
bash examples/translation_rdrop/script/run_inference.sh
```
