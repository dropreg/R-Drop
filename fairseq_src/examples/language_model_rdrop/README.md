
# Language Modeling

This example contains instructions for training a new Language Model with R-Drop.

# Prepare Data

First download and prepare the [WikiText-103 dataset](https://www.salesforce.com/products/einstein/ai-research/the-wikitext-dependency-language-modeling-dataset/):
```bash
cd examples/language_model_rdrop/
bash prepare-wikitext-103.sh
cd ../..
```

Next preprocess/binarize the data:
```bash
TEXT=examples/language_model/wikitext-103
fairseq-preprocess \
    --only-source \
    --trainpref $TEXT/wiki.train.tokens \
    --validpref $TEXT/wiki.valid.tokens \
    --testpref $TEXT/wiki.test.tokens \
    --destdir data-bin/wikitext-103 \
    --workers 20
```

# Training Script

Train a basic transformer language model with R-Drop on wikitext-103.
```
bash script/train_lm_adaptive_wiki103_rdrop.sh
```

Train a model with adaptive inputs and R-Drop using the transformer_lm_wiki103 model architecture:
```
bash script/train_lm_wiki103_rdrop.sh
```

# Inference Script

Evaluate R-Drop model over this data:
```
bash script/evaluate_lm_wiki103.sh
```
