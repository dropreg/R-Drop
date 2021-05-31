import torch
from fairseq.models.bart import BARTModel

bart = BARTModel.from_pretrained(
    '/data/lxb/cnndailymail_sum/kl-small-checkpoints/',
    checkpoint_file='checkpoint7.pt',
    data_name_or_path='/data/lxb/cnndailymail_sum/cnn_dm-bin/'
)

bart.cuda()
bart.eval()
bart.half()
count = 1
bsz = 32
with open('/data/lxb/cnndailymail_sum/cnn_dm/test.source') as source, open('/data/lxb/cnndailymail_sum/cnn_dm/test_small_kl.hypo', 'w') as fout:
    sline = source.readline().strip()
    slines = [sline]
    for sline in source:
        if count % bsz == 0:
            with torch.no_grad():
                hypotheses_batch = bart.sample(slines, beam=4, lenpen=2.0, max_len_b=140, min_len=55, no_repeat_ngram_size=3)

            for hypothesis in hypotheses_batch:
                fout.write(hypothesis + '\n')
                fout.flush()
            slines = []

        slines.append(sline.strip())
        count += 1
    if slines != []:
        hypotheses_batch = bart.sample(slines, beam=4, lenpen=2.0, max_len_b=140, min_len=55, no_repeat_ngram_size=3)
        for hypothesis in hypotheses_batch:
            fout.write(hypothesis + '\n')
            fout.flush()