wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json'
wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe'
wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/dict.txt'

TASK=cnn_dm
for SPLIT in train val
do
  for LANG in source target
  do
    python -m examples.roberta.multiprocessing_bpe_encoder \
    --encoder-json encoder.json \
    --vocab-bpe vocab.bpe \
    --inputs "/data/lxb/cnndailymail_sum/cnn_dm/$SPLIT.$LANG" \
    --outputs "/data/lxb/cnndailymail_sum/cnn_dm/$SPLIT.bpe.$LANG" \
    --workers 60 \
    --keep-empty;
  done
done


fairseq-preprocess \
  --source-lang "source" \
  --target-lang "target" \
  --trainpref "/data/lxb/cnndailymail_sum/cnn_dm/train.bpe" \
  --validpref "/data/lxb/cnndailymail_sum/cnn_dm/val.bpe" \
  --destdir "/data/lxb/cnndailymail_sum/cnn_dm-bin/" \
  --workers 60 \
  --srcdict dict.txt \
  --tgtdict dict.txt;