
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
TEXT=iwslt14.tokenized.de-en/
fairseq-preprocess \
    --source-lang en --target-lang de \
    --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
    --destdir iwslt14.tokenized.de-en/data-bin/--thresholdtgt 0 --thresholdsrc 0 \
    --workers 20
```

# Training Script

Train a Transformer translation model with R-Drop over this data:
```
bash script/run_train.sh
```

# Inference Script

Evaluate R-Drop model over this data:
```
bash script/run_inference.sh
```
