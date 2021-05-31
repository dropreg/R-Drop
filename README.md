# R-Drop: Regularized Dropout for Neural Networks

R-drop is a simple yet very effective regularization method built upon dropout, by minimizing the bidirectional KL-divergence of the output distributions of any pair of sub models sampled from dropout in model training.

R-Drop is capable to handle many tasks for both NLP and CV:

1. [Neural Machine Translation Task](fairseq_src/README.md)

2. [Abstractive Summarization Task](fairseq_src/README.md)

3. [Language Modeling Task](fairseq_src/README.md)

4. [Language Understanding Task](huggingface_transformer_src/README.md)

5. [Image Classification Task](vit_src/README.md)
