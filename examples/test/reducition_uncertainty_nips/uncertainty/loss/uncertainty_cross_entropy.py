# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import random
import math
import copy
import torch
import torch.nn as nn
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from torch.nn.modules.loss import _Loss


def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.)
        smooth_loss.masked_fill_(pad_mask, 0.)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / lprobs.size(-1)
    loss = (1. - epsilon) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss

def vanilla_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.)
    else:
        nll_loss = nll_loss.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.sum()
    return nll_loss

@register_criterion('uncertainty_label_smoothed_cross_entropy')
class UncertaintyLabelSmoothedCrossEntropyCriterion(FairseqCriterion):

    def __init__(self, task, sentence_avg, label_smoothing):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.eps = label_smoothing

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')
        # fmt: on

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample, net_output).view(-1, 1)

        loss, nll_loss = label_smoothed_nll_loss(
            lprobs, target, self.eps, ignore_index=self.padding_idx, reduce=reduce,
        )
        return loss, nll_loss
    
    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get('nll_loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)

        metrics.log_scalar('loss', loss_sum / sample_size / math.log(2), sample_size, round=3)
        metrics.log_scalar('nll_loss', nll_loss_sum / ntokens / math.log(2), ntokens, round=3)
        metrics.log_derived('ppl', lambda meters: utils.get_perplexity(meters['nll_loss'].avg))

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True

    def compute_kl_loss(self, model, p_net_output, q_net_output, pad_mask=None, reduce=True):
        p = model.get_normalized_probs(p_net_output, log_probs=True)
        p_tec = model.get_normalized_probs(p_net_output, log_probs=False)
        q = model.get_normalized_probs(q_net_output, log_probs=True)
        q_tec = model.get_normalized_probs(q_net_output, log_probs=False)

        p_loss = torch.nn.functional.kl_div(p, q_tec, reduction='none')
        q_loss = torch.nn.functional.kl_div(q, p_tec, reduction='none')
        
        if pad_mask is not None:
            p_loss.masked_fill_(pad_mask, 0.)
            q_loss.masked_fill_(pad_mask, 0.)

        if reduce:
            p_loss = p_loss.sum()
            q_loss = q_loss.sum()

        loss = (p_loss + q_loss) / 2
        return loss

    def compute_js_loss(self, model, p_net_output, q_net_output, pad_mask=None, reduce=True):
        p_tec = model.get_normalized_probs(p_net_output, log_probs=False)
        q_tec = model.get_normalized_probs(q_net_output, log_probs=False)
        p_q_tec = (p_tec + q_tec) / 2.
        p = model.get_normalized_probs(p_net_output, log_probs=True)
        q = model.get_normalized_probs(q_net_output, log_probs=True)

        p_loss = torch.nn.functional.kl_div(p, p_q_tec, reduction='none')
        q_loss = torch.nn.functional.kl_div(q, p_q_tec, reduction='none')
        
        if pad_mask is not None:
            p_loss.masked_fill_(pad_mask, 0.)
            q_loss.masked_fill_(pad_mask, 0.)

        if reduce:
            p_loss = p_loss.sum()
            q_loss = q_loss.sum()

        loss = (p_loss + q_loss) / 2
        return loss
    
    def _replace_tokens_by_predict(self, inputs, predict_sample, vocab_dict):
        bos_index, eos_index = vocab_dict.bos(), vocab_dict.eos()
        pad_index, unk_index = vocab_dict.pad(), vocab_dict.unk()
        
        predict_inputs = inputs.clone()
        predict_inputs[:,1:-1] = predict_sample[:,:-2]
        available_token_indices = (inputs != bos_index) & (inputs != eos_index) & (inputs != pad_index) & (inputs != unk_index)
        random_masking_indices = torch.bernoulli(torch.full(inputs.shape, 0.1, device=inputs.device)).bool()
        
        masking_indices = random_masking_indices & available_token_indices

        predict_inputs[~masking_indices] = 0
        masked_inputs = inputs.clone()
        masked_inputs[masking_indices] = 0
        final_inputs = masked_inputs + predict_inputs
        return final_inputs, masking_indices

    def data_augment_predict_input(self, sample, predict_sample, src_dict, tgt_dict):
        prev_out_tokens, prev_out_mask = self._replace_tokens_by_predict(sample['net_input']['prev_output_tokens'], predict_sample, tgt_dict)
        da_input =  {
            'src_tokens': self._mask_tokens_by_word(sample['net_input']['src_tokens'], src_dict)[0],
            'src_lengths': sample['net_input']['src_lengths'].clone(),
            'prev_output_tokens':  prev_out_tokens,
        }
        return da_input, prev_out_mask

    def _mask_tokens_by_word(self, inputs, vocab_dict):
        bos_index, eos_index = vocab_dict.bos(), vocab_dict.eos()
        pad_index, unk_index = vocab_dict.pad(), vocab_dict.unk()
        
        available_token_indices = (inputs != bos_index) & (inputs != eos_index) & (inputs != pad_index) & (inputs != unk_index)
        random_masking_indices = torch.bernoulli(torch.full(inputs.shape, 0.05, device=inputs.device)).bool()

        mask_target = inputs.clone()
        masked_inputs = inputs.clone()
        masking_indices = random_masking_indices & available_token_indices
        masked_inputs[masking_indices] = unk_index
        return masked_inputs, masking_indices, mask_target

    def data_augment_input(self, sample, src_dict, tgt_dict):
        da_input =  {
            'src_tokens': self._mask_tokens_by_word(sample['net_input']['src_tokens'], src_dict)[0],
            'src_lengths': sample['net_input']['src_lengths'].clone(),
            'prev_output_tokens': self._mask_tokens_by_word(sample['net_input']['prev_output_tokens'], tgt_dict)[0],
        }
        return da_input
    
    def forward_kl(self, model, sample, src_dict, tgt_dict, optimizer, reduce=True):
        
        # for base arch
        model.reset_space()
        base_net_output = model(**sample['net_input'])
        base_loss, base_nll_loss = self.compute_loss(model, base_net_output, sample, reduce=reduce)
        targets = model.get_targets(sample, base_net_output).unsqueeze(-1)
        pad_mask = targets.eq(self.padding_idx)
        
        # model.sample_dropout_space()
        model.sample_reorder_space_by_probs(0.1)
        variant_net_output = model(**sample['net_input'])
        variant_loss, variant_nll_loss = self.compute_loss(model, variant_net_output, sample, reduce=reduce)

        # for kl constriant
        kl_loss = self.compute_kl_loss(model, base_net_output, variant_net_output, pad_mask)
        
        loss = base_loss + variant_loss + 5 * kl_loss
        nll_loss = base_nll_loss + variant_nll_loss
        optimizer.backward(loss)

        ntokens = sample['ntokens']
        nsentences = sample['target'].size(0)
        sample_size = sample['ntokens']
        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'nll_loss': utils.item(nll_loss.data) if reduce else nll_loss.data,
            'kl_loss': utils.item(kl_loss.data) if reduce else kl_loss.data,
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        model.reset_space()
        net_output = model(**sample['net_input'])
        loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
        sample_size = sample['target'].size(0) if self.sentence_avg else sample['ntokens']
        logging_output = {
            'loss': loss.data,
            'nll_loss': nll_loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output