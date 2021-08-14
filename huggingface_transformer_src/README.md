# Introduction

This Repo conduct language understanding task.

# Requirements and Installation

* [PyTorch](http://pytorch.org/) version == 1.8
* Python version >= 3.6

To install fairseq from source and develop locally:
```
git clone https://github.com/dropreg/R-Drop.git
cd R-Drop/huggingface_transformer/
pip install --editable .
```

# Fine-tuning on GLUE task:
Example fine-tuning cmd for `MRPC` task

```bash
cd bert_rdrop/
bash run.sh
```

You can run it on Roberta by simply switching hyper-parameter: model_name_or_path

```bash
--model_name_or_path roberta-base \
```

For each of the GLUE task, you will need to use following cmd-line arguments:

Model | MNLI | QNLI | QQP | RTE | SST-2 | MRPC | CoLA | STS-B
---|---|---|---|---|---|---|---|---
`--lr` | 1e-5 | 1e-5 | 1e-5 | 1e-5 | 1e-5 | 1e-5 | 1e-5 | 1e-5
`--batch-size` | 32 | 32 | 32 | 8 | 32 | 16 | 16 | 16
`--total-num-update` | 123873 | 33112 | 113272 | 2036 | 20935 | 2296 | 5336 | 3598
`--warmup-updates` | 7432 | 1986 | 28318 | 122 | 1256 | 137 | 320 | 214
