from fairseq.models.roberta import RobertaModel
import torch

one = torch.ones([2, 2])
zero = torch.zeros([2, 2])
print(torch.stack([one, zero]))
print(torch.stack([one, zero]).view(2, -1))
print(torch.stack([one, zero]).view(2, -1).view(2, -1))
exit()

roberta = RobertaModel.from_pretrained(
    '/data/lxb/glue_result/roberta/MRPC/',
    checkpoint_file='checkpoint10.pt',
    data_name_or_path='/data/lxb/glue_result/data/MRPC-bin'
)

label_fn = lambda label: roberta.task.label_dictionary.string(
    [label + roberta.task.label_dictionary.nspecial]
)

for idx, l in enumerate(roberta.model.encoder.sentence_encoder.layers):
    l.dropout_module.p = 0.2
    l.dropout_module.apply_during_inference = True

ncorrect, nsamples = 0, 0
roberta.cuda()
roberta.eval()
with open('/data/lxb/glue_result/data/glue_data/MRPC/dev.tsv') as fin:
    fin.readline()
    for index, line in enumerate(fin):
        tokens = line.strip().split('\t')
        sent1, sent2, target = tokens[3], tokens[4], tokens[0]
        tokens = roberta.encode(sent1, sent2)
        prediction = roberta.predict('sentence_classification_head', tokens).argmax().item()
        prediction_label = label_fn(prediction)
        ncorrect += int(prediction_label == target)
        nsamples += 1
print('| Accuracy: ', float(ncorrect)/float(nsamples))