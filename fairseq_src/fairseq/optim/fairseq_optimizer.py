# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from fairseq import utils
from fairseq.dataclass.utils import gen_parser_from_dataclass
import numpy as np


class FairseqOptimizer(object):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.surgery_grad_cache = {}
        self.surgery_shape_cache = {}

    @classmethod
    def add_args(cls, parser):
        """Add optimizer-specific arguments to the parser."""
        dc = getattr(cls, "__dataclass", None)
        if dc is not None:
            gen_parser_from_dataclass(parser, dc())

    @property
    def optimizer(self):
        """Return a torch.optim.optimizer.Optimizer instance."""
        if not hasattr(self, "_optimizer"):
            raise NotImplementedError
        if not isinstance(self._optimizer, torch.optim.Optimizer):
            raise ValueError("_optimizer must be an instance of torch.optim.Optimizer")
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        """Reset optimizer instance."""
        if not hasattr(self, "_optimizer"):
            raise NotImplementedError
        if not isinstance(self._optimizer, torch.optim.Optimizer):
            raise ValueError("_optimizer must be an instance of torch.optim.Optimizer")
        self._optimizer = optimizer

    @property
    def optimizer_config(self):
        """
        Return a kwarg dictionary that will be used to override optimizer
        args stored in checkpoints. This allows us to load a checkpoint and
        resume training using a different set of optimizer args, e.g., with a
        different learning rate.
        """
        raise NotImplementedError

    @property
    def params(self):
        """Return an iterable of the parameters held by the optimizer."""
        for param_group in self.param_groups:
            for p in param_group["params"]:
                yield p

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    def __getstate__(self):
        return self._optimizer.__getstate__()

    def get_lr(self):
        """Return the current learning rate."""
        return self.param_groups[0]["lr"]

    def set_lr(self, lr):
        """Set the learning rate."""
        for param_group in self.param_groups:
            param_group["lr"] = lr

    def state_dict(self):
        """Return the optimizer's state dict."""
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict, optimizer_overrides=None):
        """Load an optimizer state dict.

        In general we should prefer the configuration of the existing optimizer
        instance (e.g., learning rate) over that found in the state_dict. This
        allows us to resume training from a checkpoint using a new set of
        optimizer args.
        """
        self.optimizer.load_state_dict(state_dict)

        if optimizer_overrides is not None and len(optimizer_overrides) > 0:
            # override learning rate, momentum, etc. with latest values
            for group in self.param_groups:
                group.update(optimizer_overrides)

    def backward(self, loss):
        """Computes the sum of gradients of the given tensor w.r.t. graph leaves."""
        loss.backward()

    def all_reduce_grads(self, module):
        """Manually all-reduce gradients (if required)."""
        if hasattr(module, "all_reduce_grads"):
            module.all_reduce_grads()

    def multiply_grads(self, c):
        """Multiplies grads by a constant *c*."""
        for p in self.params:
            if p.grad is not None:
                if torch.is_tensor(c):
                    c = c.to(p.grad.device)
                p.grad.data.mul_(c)

    def clip_grad_norm(self, max_norm, aggregate_norm_fn=None):
        """Clips gradient norm."""
        return utils.clip_grad_norm_(self.params, max_norm, aggregate_norm_fn)

    def step(self, closure=None, scale=1.0, groups=None):
        """Performs a single optimization step."""
        if self.supports_step_with_scale:
            if self.supports_groups:
                self.optimizer.step(closure, scale=scale, groups=groups)
            else:
                self.optimizer.step(closure, scale=scale)
        else:
            if scale != 1.0:
                self.multiply_grads(1.0 / scale)
            if self.supports_groups:
                self.optimizer.step(closure, groups=groups)
            else:
                self.optimizer.step(closure)

    def zero_grad(self):
        """Clears the gradients of all optimized parameters."""
        for p in self.params:
            p.grad = None
        self.optimizer.zero_grad()

    @property
    def supports_memory_efficient_fp16(self):
        if hasattr(self.optimizer, "supports_memory_efficient_fp16"):
            return self.optimizer.supports_memory_efficient_fp16
        return False

    @property
    def supports_step_with_scale(self):
        if hasattr(self.optimizer, "supports_step_with_scale"):
            return self.optimizer.supports_step_with_scale
        return False

    @property
    def supports_groups(self):
        if hasattr(self.optimizer, "supports_groups"):
            return self.optimizer.supports_groups
        return False

    @property
    def supports_flat_params(self):
        """
        Whether the optimizer supports collapsing of the model
        parameters/gradients into a single contiguous Tensor.
        """
        if hasattr(self.optimizer, "supports_flat_params"):
            return self.optimizer.supports_flat_params
        return False

    def average_params(self):
        pass

    def broadcast_global_state_dict(self, state_dict):
        """
        Broadcasts a global state dict to all ranks.
        Useful for optimizers that shard state between ranks.
        """
        if hasattr(self.optimizer, "broadcast_global_state_dict"):
            return self.optimizer.broadcast_global_state_dict(state_dict)
        else:
            return state_dict

    def zero_surgery_cache(self):
        self.surgery_grad_cache = {}
        self.surgery_encoder_grad_cache = {}
        self.surgery_shape_cache = {}

    def save_surgery_grad(self, surgery_key):
        """ must run after loss backward """
        if surgery_key not in self.surgery_grad_cache:
            self.surgery_grad_cache[surgery_key] = []
            self.surgery_shape_cache[surgery_key] = []

        grad_tmp = []
        if getattr(self, "fp32_optimizer", None) is not None:
            for p in self.fp16_params:
                if p.grad is None: continue
                grad_tmp.append(p.grad.clone())
                self.surgery_shape_cache[surgery_key].append(p.grad.shape)
            self.surgery_grad_cache[surgery_key] = torch.cat([g.flatten() for g in grad_tmp]).double()
            self.surgery_encoder_grad_cache[surgery_key] = torch.cat([g.flatten() for g in grad_tmp[len(grad_tmp)//2:]]).double()
        else:
            for group in self.optimizer.param_groups:
                for p in group["params"]:
                    if p.grad is None: continue
                    grad_tmp.append(p.grad.clone())
                    self.surgery_shape_cache[surgery_key].append(p.grad.shape)
            self.surgery_grad_cache[surgery_key] = torch.cat([g.flatten() for g in grad_tmp])
            self.surgery_encoder_grad_cache[surgery_key] = torch.cat([g.flatten() for g in grad_tmp[len(grad_tmp)//2:]])
    
    def set_surgery_grad(self, surgery_all_key):
        for surgery_key in surgery_all_key:
            unflatten_grad, idx = [], 0
            for shape in self.surgery_shape_cache[surgery_key]:
                length = np.prod(shape)
                unflatten_grad.append(self.surgery_grad_cache[surgery_key][idx:idx + length].view(shape))
                idx += length
            p_idx = 0
            for group in self.optimizer.param_groups:
                if getattr(self, "fp32_optimizer", None) is not None:
                    for p in self.fp16_params:
                        if p.grad is None: continue
                        p.grad += unflatten_grad[p_idx]
                        p_idx += 1
                else:
                    for p in group['params']:
                        if p.grad is None: continue
                        p.grad += unflatten_grad[p_idx]
                        p_idx += 1

    def gradient_sim(self, master_key, comp_key):
        g_i = self.surgery_grad_cache[master_key]
        g_j = self.surgery_grad_cache[comp_key]
        # g_i = self.surgery_encoder_grad_cache[master_key]
        # g_j = self.surgery_encoder_grad_cache[comp_key]
        cosine = torch.cosine_similarity(g_i, g_j, dim=0)
        return cosine


class LegacyFairseqOptimizer(FairseqOptimizer):
    def __init__(self, args):
        self.args = args
