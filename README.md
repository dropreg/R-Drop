# R-Drop: Regularized Dropout for Neural Networks

This repo contains the code of our NeurIPS-2021 paper, [R-drop: Regularized Dropout for Neural Networks](https://arxiv.org/abs/2106.14448). 

R-Drop is a simple yet very effective regularization method built upon dropout, by minimizing the bidirectional KL-divergence of the output distributions of any pair of sub models sampled from dropout in model training.

```
@inproceedings{liang2021rdrop,
  title={R-Drop: Regularized Dropout for Neural Networks},
  author={Liang, Xiaobo* and Wu, Lijun* and Li, Juntao and Wang, Yue and Meng, Qi and Qin, Tao and Chen, Wei and Zhang, Min and Liu, Tie-Yan},
  booktitle={NeurIPS},
  year={2021}
}
```

##  Usage:
R-Drop is an almost universal method for supervised tasks and even performs well for semi-supervised setting. 
For other settings and tasks that are not mentioned in our paper, feel free to try the following piece of code.

```python
import torch.nn.functional as F

# define your task model, which outputs the classifier logits
model = TaskModel()

def compute_kl_loss(self, p, q, pad_mask=None):
    
    p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='none')
    q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction='none')
    
    # pad_mask is for seq-level tasks
    if pad_mask is not None:
        p_loss.masked_fill_(pad_mask, 0.)
        q_loss.masked_fill_(pad_mask, 0.)

    # You can choose whether to use function "sum" and "mean" depending on your task
    p_loss = p_loss.sum()
    q_loss = q_loss.sum()

    loss = (p_loss + q_loss) / 2
    return loss

# keep dropout and forward twice
logits = model(x)

logits2 = model(x)

# cross entropy loss for classifier
ce_loss = 0.5 * (cross_entropy_loss(logits, label) + cross_entropy_loss(logits2, label))

kl_loss = compute_kl_loss(logits, logits2)

# carefully choose hyper-parameters
loss = ce_loss + α * kl_loss

```

## Quick Links:
R-Drop is capable to handle many tasks for both NLP and CV:

1. [Neural Machine Translation Task](fairseq_src/README.md)

2. [Abstractive Summarization Task](fairseq_src/README.md)

3. [Language Modeling Task](fairseq_src/README.md)

4. [Language Understanding Task](huggingface_transformer_src/README.md)

5. [Image Classification Task](vit_src/README.md)



