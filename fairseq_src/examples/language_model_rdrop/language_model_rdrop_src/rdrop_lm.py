import torch
import logging
from fairseq.tasks import register_task
from fairseq.tasks.language_modeling import LanguageModelingTask, LanguageModelingConfig
from fairseq import metrics, utils
import numpy as np


logger = logging.getLogger(__name__)


@register_task("rdrop_lm", dataclass=LanguageModelingConfig)
class RDropLM(LanguageModelingTask):

    def __init__(self, args, dictionary, output_dictionary=None, targets=None):
        super().__init__(args, dictionary, output_dictionary, targets)
    
    def build_model(self, args):
        model = super().build_model(args)
        return model
    
    def train_step(self, sample, model, criterion, optimizer, update_num, ignore_grad=False):
        model.train()
        model.set_num_updates(update_num)
        with torch.autograd.profiler.record_function("forward"):
            loss, sample_size, logging_output = criterion.forward_reg(model, sample)
            if ignore_grad:
                loss *= 0
            with torch.autograd.profiler.record_function("backward"):
                optimizer.backward(loss)
            return loss, sample_size, logging_output

        