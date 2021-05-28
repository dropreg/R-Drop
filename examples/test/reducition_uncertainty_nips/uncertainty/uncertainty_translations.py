import torch
import logging
from fairseq.tasks import register_task
from fairseq.tasks.translation import TranslationTask
from fairseq import metrics, utils
import numpy as np


logger = logging.getLogger(__name__)

@register_task("uncertainty_translation")
class UncertaintyTranslation(TranslationTask):

    @staticmethod
    def add_args(parser):
        TranslationTask.add_args(parser)

    def __init__(self, args, src_dict, tgt_dict):
        super().__init__(args, src_dict, tgt_dict)
    
    def build_model(self, args):
        model = super().build_model(args)
        return model
    
    def train_step(self, sample, model, criterion, optimizer, update_num, ignore_grad=False):
        model.train()
        model.set_num_updates(update_num)
        with torch.autograd.profiler.record_function("forward"):
            loss, sample_size, logging_output = criterion.forward_kl(model, sample, self.src_dict, self.tgt_dict, optimizer)
            return loss, sample_size, logging_output

    def valid_step(self, sample, model, criterion):
        model.reset_space()
        loss, sample_size, logging_output = super().valid_step(sample, model, criterion)
        return loss, sample_size, logging_output
    
    def inference_step(
        self, generator, models, sample, prefix_tokens=None, constraints=None
    ):
        with torch.no_grad():
            models[0].reset_space()
            return generator.generate(
                models,
                sample,
                prefix_tokens=prefix_tokens,
                constraints=constraints,
            )
    
    def reduce_metrics(self, logging_outputs, criterion):
        super().reduce_metrics(logging_outputs, criterion)

        if len(logging_outputs) > 0 and "kl_loss" in logging_outputs[0]:
            metrics.log_scalar('kl_loss', sum(l['kl_loss'] for l in logging_outputs))
        