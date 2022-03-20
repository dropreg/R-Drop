# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from fairseq.dataclass.constants import DDP_BACKEND_CHOICES
from omegaconf import II


@dataclass
class AdaptiveLossConfig(FairseqDataclass):
    sentence_avg: bool = II("optimization.sentence_avg")
    ddp_backend: DDP_BACKEND_CHOICES = II("distributed_training.ddp_backend")
    reg_alpha: float = field(
        default=1.0,
        metadata={"help": "weight for kl loss, default is 1.0"},
    )


@register_criterion("reg_adaptive_loss", dataclass=AdaptiveLossConfig)
class RegAdaptiveLoss(FairseqCriterion):
    """This is an implementation of the loss function accompanying the adaptive softmax approximation for
    graphical processing units (GPU), described in the paper "Efficient softmax approximation for GPUs"
    (http://arxiv.org/abs/1609.04309)."""

    def __init__(self, task, sentence_avg, reg_alpha):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.reg_alpha = reg_alpha
    
    @classmethod
    def build_criterion(cls, cfg: AdaptiveLossConfig, task):
        if cfg.ddp_backend in {"c10d", "pytorch_ddp"}:
            raise Exception(
                "AdaptiveLoss is not compatible with the PyTorch "
                "version of DistributedDataParallel. Please use "
                "`--ddp-backend=legacy_ddp` instead."
            )
        return cls(task, cfg.sentence_avg, cfg.reg_alpha)

    def forward_reg(self, model, sample, reduce=True):
        
        assert (
            hasattr(model.decoder, "adaptive_softmax")
            and model.decoder.adaptive_softmax is not None
        )
        adaptive_softmax = model.decoder.adaptive_softmax

        net_output = model(**sample["net_input"])
        orig_target = model.get_targets(sample, net_output)

        nsentences = orig_target.size(0) 
        orig_target = orig_target.view(-1)

        bsz = orig_target.size(0)

        logits, target = adaptive_softmax(net_output[0], orig_target)
        assert len(target) == len(logits)

        another_net_output = model(**sample["net_input"])
        another_logits, _ = adaptive_softmax(another_net_output[0], orig_target)
        # target should be the above same target
        assert len(another_logits) == len(target)

        base_loss = net_output[0].new(1 if reduce else bsz).zero_()
        another_loss = torch.zeros(base_loss.shape).cuda()
        loss = torch.zeros(base_loss.shape).cuda()
        kl_loss = torch.tensor(0.).cuda()

        for i in range(len(target)):
            if target[i] is not None:
                assert target[i].min() >= 0 and target[i].max() <= logits[i].size(1)
                base_loss += F.cross_entropy(
                    logits[i],
                    target[i],
                    ignore_index=self.padding_idx,
                    reduction="sum" if reduce else "none",
                )
                another_loss += F.cross_entropy(
                    another_logits[i], 
                    target[i],
                    ignore_index=self.padding_idx,
                    reduction="sum" if reduce else "none",
                )
                # compute kl_los, default reduce is True
                pad_mask = target[i].eq(self.padding_idx).unsqueeze(-1)
                p_log_softmax = F.log_softmax(logits[i], dim=-1)
                p_softmax = F.softmax(logits[i], dim=-1)
                q_log_softmax = F.log_softmax(another_logits[i], dim=-1)
                q_softmax = F.softmax(another_logits[i], dim=-1)
                p_loss = F.kl_div(p_log_softmax, q_softmax, reduction='none')
                q_loss = F.kl_div(q_log_softmax, p_softmax, reduction='none')
                if pad_mask is not None:
                    p_loss.masked_fill_(pad_mask, 0.)
                    q_loss.masked_fill_(pad_mask, 0.)
                if reduce:
                    p_loss = p_loss.sum()
                    q_loss = q_loss.sum()
                kl_loss += (p_loss + q_loss) / 2.0

        loss = base_loss + another_loss
        loss += self.kl_weight * kl_loss
        
        orig = utils.strip_pad(orig_target, self.padding_idx)
        ntokens = orig.numel()
        sample_size = sample["target"].size(0) if self.sentence_avg else ntokens
        logging_output = {
            "loss": loss.data,
            "ntokens": ntokens,
            'kl_loss': kl_loss.data,
            "nsentences": nsentences,
            "sample_size": sample_size,
        }
        return loss, sample_size, logging_output

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """

        assert (
            hasattr(model.decoder, "adaptive_softmax")
            and model.decoder.adaptive_softmax is not None
        )
        adaptive_softmax = model.decoder.adaptive_softmax

        net_output = model(**sample["net_input"])
        orig_target = model.get_targets(sample, net_output)

        nsentences = orig_target.size(0)
        orig_target = orig_target.view(-1)

        bsz = orig_target.size(0)

        logits, target = adaptive_softmax(net_output[0], orig_target)
        assert len(target) == len(logits)

        loss = net_output[0].new(1 if reduce else bsz).zero_()

        for i in range(len(target)):
            if target[i] is not None:
                assert target[i].min() >= 0 and target[i].max() <= logits[i].size(1)
                loss += F.cross_entropy(
                    logits[i],
                    target[i],
                    ignore_index=self.padding_idx,
                    reduction="sum" if reduce else "none",
                )

        orig = utils.strip_pad(orig_target, self.padding_idx)
        ntokens = orig.numel()
        sample_size = sample["target"].size(0) if self.sentence_avg else ntokens
        logging_output = {
            "loss": loss.data,
            "ntokens": ntokens,
            "nsentences": nsentences,
            "sample_size": sample_size,
        }
        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = utils.item(sum(log.get("loss", 0) for log in logging_outputs))
        ntokens = utils.item(sum(log.get("ntokens", 0) for log in logging_outputs))
        sample_size = utils.item(
            sum(log.get("sample_size", 0) for log in logging_outputs)
        )

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        if sample_size != ntokens:
            metrics.log_scalar(
                "nll_loss", loss_sum / ntokens / math.log(2), ntokens, round=3
            )
            metrics.log_derived(
                "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
            )
        else:
            metrics.log_derived(
                "ppl", lambda meters: utils.get_perplexity(meters["loss"].avg)
            )
        
        if len(logging_outputs) > 0 and "kl_loss" in logging_outputs[0]:
            metrics.log_scalar('kl_loss', sum(l['kl_loss'] for l in logging_outputs))
        

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
