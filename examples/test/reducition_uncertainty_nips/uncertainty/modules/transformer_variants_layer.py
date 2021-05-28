from typing import Dict, List, Optional
import torch
import torch.nn as nn
from fairseq import utils
from fairseq.modules import LayerNorm, MultiheadAttention
from fairseq.modules.quant_noise import quant_noise
from fairseq.modules.fairseq_dropout import FairseqDropout
from torch import Tensor


class TransformerEncoderSublayer(nn.Module):

    def __init__(self, layer_arch, args):
        super().__init__()
        self.layer_prefix, self.layer_suffix = layer_arch.split("-")

        self.args = args
        self.embed_dim = args.encoder_embed_dim
        self.quant_noise = getattr(args, 'quant_noise_pq', 0)
        self.quant_noise_block_size = getattr(args, 'quant_noise_pq_block_size', 8) or 8
        self.normalize_before = args.encoder_normalize_before
        self.dropout_module = FairseqDropout(
            args.dropout, module_name=self.__class__.__name__
        )
        if self.layer_prefix == 's':
            self.self_attn = self.build_self_attention(self.embed_dim, args)
            self.self_attn_layer_norm = LayerNorm(self.embed_dim)

        else:

            self.activation_fn = utils.get_activation_fn(
                activation=getattr(args, 'activation_fn', 'relu') or "relu"
            )
            activation_dropout_p = getattr(args, "activation_dropout", 0) or 0
            if activation_dropout_p == 0:
                # for backwards compatibility with models that use args.relu_dropout
                activation_dropout_p = getattr(args, "relu_dropout", 0) or 0
            self.activation_dropout_module = FairseqDropout(
                float(activation_dropout_p), module_name=self.__class__.__name__
            )
            self.fc1 = self.build_fc1(
                self.embed_dim,
                args.encoder_ffn_embed_dim,
                self.quant_noise,
                self.quant_noise_block_size,
            )
            self.fc2 = self.build_fc2(
                args.encoder_ffn_embed_dim,
                self.embed_dim,
                self.quant_noise,
                self.quant_noise_block_size,
            )

            self.final_layer_norm = LayerNorm(self.embed_dim)

    def build_fc1(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(
            nn.Linear(input_dim, output_dim), p=q_noise, block_size=qn_block_size
        )

    def build_fc2(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(
            nn.Linear(input_dim, output_dim), p=q_noise, block_size=qn_block_size
        )

    def build_self_attention(self, embed_dim, args):
        return MultiheadAttention(
            embed_dim,
            args.encoder_attention_heads,
            dropout=args.attention_dropout,
            self_attention=True,
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
        )

    def residual_connection(self, x, residual):
        return residual + x

    def upgrade_state_dict_named(self, state_dict, name):
        """
        Rename layer norm states from `...layer_norms.0.weight` to
        `...self_attn_layer_norm.weight` and `...layer_norms.1.weight` to
        `...final_layer_norm.weight`
        """
        layer_norm_map = {"0": "self_attn_layer_norm", "1": "final_layer_norm"}
        for old, new in layer_norm_map.items():
            for m in ("weight", "bias"):
                k = "{}.layer_norms.{}.{}".format(name, old, m)
                if k in state_dict:
                    state_dict["{}.{}.{}".format(name, new, m)] = state_dict[k]
                    del state_dict[k]

    def forward(self, x, encoder_padding_mask, attn_mask: Optional[Tensor] = None):

        if attn_mask is not None:
            attn_mask = attn_mask.masked_fill(attn_mask.to(torch.bool), -1e8)
        
        if self.layer_prefix == "s":
            residual = x
            if self.normalize_before:
                x = self.self_attn_layer_norm(x)
            x, _ = self.self_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=encoder_padding_mask,
                attn_mask=attn_mask,
            )
            x = self.dropout_module(x)
            x = self.residual_connection(x, residual)
            if not self.normalize_before:
                x = self.self_attn_layer_norm(x)
            return x
        else:
            residual = x
            if self.normalize_before:
                x = self.final_layer_norm(x)
            x = self.activation_fn(self.fc1(x))
            x = self.activation_dropout_module(x)
            x = self.fc2(x)
            x = self.dropout_module(x)
            x = self.residual_connection(x, residual)
            if not self.normalize_before:
                x = self.final_layer_norm(x)
            return x


class TransformerDecoderSublayer(nn.Module):

    def __init__(
        self, args, layer_arch, no_encoder_attn=False, add_bias_kv=False, add_zero_attn=False
    ):
        super().__init__()
        self.layer_prefix, self.layer_suffix = layer_arch.split("-")

        self.embed_dim = args.decoder_embed_dim
        self.dropout_module = FairseqDropout(args.dropout, module_name=self.__class__.__name__)
        self.quant_noise = getattr(args, "quant_noise_pq", 0)
        self.quant_noise_block_size = getattr(args, "quant_noise_pq_block_size", 8)
        self.cross_self_attention = getattr(args, "cross_self_attention", False)

        self.normalize_before = args.decoder_normalize_before
        export = getattr(args, "char_inputs", False)

        if self.layer_prefix == "s":
            self.self_attn = self.build_self_attention(
                self.embed_dim,
                args,
                add_bias_kv=add_bias_kv,
                add_zero_attn=add_zero_attn,
            )
            self.self_attn_layer_norm = LayerNorm(self.embed_dim, export=export)

        if self.layer_prefix == "c":
            if no_encoder_attn:
                self.encoder_attn = None
                self.encoder_attn_layer_norm = None
            else:
                self.encoder_attn = self.build_encoder_attention(self.embed_dim, args)
                self.encoder_attn_layer_norm = LayerNorm(self.embed_dim, export=export)

        if self.layer_prefix == "f":  
            self.activation_fn = utils.get_activation_fn(
                activation=str(args.activation_fn) if getattr(args, "activation_fn", None) is not None else "relu"
            )
            activation_dropout_p = getattr(args, "activation_dropout", 0)
            if activation_dropout_p == 0:
                # for backwards compatibility with models that use args.relu_dropout
                activation_dropout_p = getattr(args, "relu_dropout", 0)
            self.activation_dropout_module = FairseqDropout(
                float(activation_dropout_p), module_name=self.__class__.__name__)
            self.fc1 = self.build_fc1(
                self.embed_dim, args.decoder_ffn_embed_dim, self.quant_noise, self.quant_noise_block_size
            )
            self.fc2 = self.build_fc2(
                args.decoder_ffn_embed_dim, self.embed_dim, self.quant_noise, self.quant_noise_block_size
            )
            self.final_layer_norm = LayerNorm(self.embed_dim, export=export)

        self.need_attn = True
        self.onnx_trace = False

    def build_fc1(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(nn.Linear(input_dim, output_dim), q_noise, qn_block_size)

    def build_fc2(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(nn.Linear(input_dim, output_dim), q_noise, qn_block_size)

    def build_self_attention(self, embed_dim, args, add_bias_kv=False, add_zero_attn=False):
        return MultiheadAttention(
            embed_dim,
            args.decoder_attention_heads,
            dropout=args.attention_dropout,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
            self_attention=not getattr(args, "cross_self_attention", False),
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
        )

    def build_encoder_attention(self, embed_dim, args):
        return MultiheadAttention(
            embed_dim,
            args.decoder_attention_heads,
            kdim=getattr(args, "encoder_embed_dim", None),
            vdim=getattr(args, "encoder_embed_dim", None),
            dropout=args.attention_dropout,
            encoder_decoder_attention=True,
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
        )

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def forward(
        self,
        x,
        encoder_out: Optional[torch.Tensor] = None,
        encoder_padding_mask: Optional[torch.Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        prev_self_attn_state: Optional[List[torch.Tensor]] = None,
        prev_attn_state: Optional[List[torch.Tensor]] = None,
        self_attn_mask: Optional[torch.Tensor] = None,
        self_attn_padding_mask: Optional[torch.Tensor] = None,
        need_attn: bool = False,
        need_head_weights: bool = False,
    ):
        if need_head_weights:
            need_attn = True

        if self.layer_prefix == "s":
            residual = x
            if self.normalize_before:
                x = self.self_attn_layer_norm(x)

            if prev_self_attn_state is not None:
                prev_key, prev_value = prev_self_attn_state[:2]
                saved_state: Dict[str, Optional[Tensor]] = {
                    "prev_key": prev_key,
                    "prev_value": prev_value,
                }
                if len(prev_self_attn_state) >= 3:
                    saved_state["prev_key_padding_mask"] = prev_self_attn_state[2]
                assert incremental_state is not None
                self.self_attn._set_input_buffer(incremental_state, saved_state)
            _self_attn_input_buffer = self.self_attn._get_input_buffer(incremental_state)

            if self.cross_self_attention and not (
                incremental_state is not None
                and _self_attn_input_buffer is not None
                and "prev_key" in _self_attn_input_buffer
            ):
                if self_attn_mask is not None:
                    assert encoder_out is not None
                    self_attn_mask = torch.cat(
                        (x.new_zeros(x.size(0), encoder_out.size(0)), self_attn_mask), dim=1
                    )
                if self_attn_padding_mask is not None:
                    if encoder_padding_mask is None:
                        assert encoder_out is not None
                        encoder_padding_mask = self_attn_padding_mask.new_zeros(
                            encoder_out.size(1), encoder_out.size(0)
                        )
                    self_attn_padding_mask = torch.cat(
                        (encoder_padding_mask, self_attn_padding_mask), dim=1
                    )
                assert encoder_out is not None
                y = torch.cat((encoder_out, x), dim=0)
            else:
                y = x

            x, attn = self.self_attn(
                query=x,
                key=y,
                value=y,
                key_padding_mask=self_attn_padding_mask,
                incremental_state=incremental_state,
                need_weights=False,
                attn_mask=self_attn_mask,
            )
            x = self.dropout_module(x)
            x = residual + x
            if not self.normalize_before:
                x = self.self_attn_layer_norm(x)

            return x, attn, None

        elif self.layer_prefix ==  "c":
            residual = x
            if self.normalize_before:
                x = self.encoder_attn_layer_norm(x)
            if prev_attn_state is not None:
                prev_key, prev_value = prev_attn_state[:2]
                saved_state: Dict[str, Optional[Tensor]] = {
                    "prev_key": prev_key,
                    "prev_value": prev_value,
                }
                if len(prev_attn_state) >= 3:
                    saved_state["prev_key_padding_mask"] = prev_attn_state[2]
                assert incremental_state is not None
                self.encoder_attn._set_input_buffer(incremental_state, saved_state)

            x, attn = self.encoder_attn(
                query=x,
                key=encoder_out,
                value=encoder_out,
                key_padding_mask=encoder_padding_mask,
                incremental_state=incremental_state,
                static_kv=True,
                need_weights=need_attn or (not self.training and self.need_attn),
                need_head_weights=need_head_weights,
            )
            x = self.dropout_module(x)
            x = residual + x
            if not self.normalize_before:
                x = self.encoder_attn_layer_norm(x)

            return x, attn, None

        elif self.layer_prefix ==  "f":
            residual = x
            if self.normalize_before:
                x = self.final_layer_norm(x)

            x = self.activation_fn(self.fc1(x))
            x = self.activation_dropout_module(x)
            x = self.fc2(x)
            x = self.dropout_module(x)
            x = residual + x
            if not self.normalize_before:
                x = self.final_layer_norm(x)

            if self.onnx_trace and incremental_state is not None:
                saved_state = self.self_attn._get_input_buffer(incremental_state)
                assert saved_state is not None
                if self_attn_padding_mask is not None:
                    self_attn_state = [
                        saved_state["prev_key"],
                        saved_state["prev_value"],
                        saved_state["prev_key_padding_mask"],
                    ]
                else:
                    self_attn_state = [saved_state["prev_key"], saved_state["prev_value"]]
                return x, attn, self_attn_state

            return x, None, None
        else:
            raise EnvironmentError("TransformerDecoderSublayer not support current sublayer prefix")

    def make_generation_fast_(self, need_attn: bool = False, **kwargs):
        self.need_attn = need_attn
