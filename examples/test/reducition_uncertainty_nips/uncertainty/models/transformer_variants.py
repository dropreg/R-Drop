from fairseq.models import register_model, register_model_architecture
from fairseq.models.transformer import TransformerModel, TransformerDecoder, TransformerEncoder, base_architecture
from ..modules.transformer_variants_layer import TransformerDecoderSublayer, TransformerEncoderSublayer
import numpy as np
import torch.nn as nn
from fairseq.modules import LayerDropModuleList
from typing import Any, Dict, List, Optional, Tuple
from torch import Tensor
import torch
import logging


logger = logging.getLogger(__name__)

@register_model("variants_transformer")
class VariantsTransformerModel(TransformerModel):

    def __init__(self, args, encoder, decoder):
        super().__init__(args, encoder, decoder)

    def reset_space(self):
        self.decoder.reset_space()

    def sample_reorder_space_by_probs(self, p):
        self.decoder.reorder_by_probs(p)
    
    def sample_dropout_space(self, log_flag=False):
        if log_flag:
            print("before decoder drop: {}".format(self.decoder.get_space()))
        self.decoder.drop_current_layer()
        if log_flag:
            print("after decoder drop: {}".format(self.decoder.get_space()))

    def sample_reorder_space(self, log_flag=False):
        if log_flag:
            print("before decoder reorder: {}".format(self.decoder.get_space()))
        self.decoder.reorder_current_order()
        if log_flag:
            print("after decoder reorder: {}".format(self.decoder.get_space()))

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        return TransformerDecoderVariants(args, tgt_dict, embed_tokens, 
                no_encoder_attn=getattr(args, "no_cross_attention", False),)

class TransformerDecoderVariants(TransformerDecoder):

    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
        super().__init__( args, dictionary, embed_tokens, no_encoder_attn)
        
        self._num_updates = 0
        # self.layers = nn.ModuleList([])
        self.layers = LayerDropModuleList([])
        decoder_layers, self.vanilla_order = self.build_sublayers(args, no_encoder_attn)
        self.layers.extend(decoder_layers)
        self.reorder_space = self.reorder_search_space()
        self.decoder_current_order = self.vanilla_order
    
    def build_sublayers(self, args, no_encoder_attn=False):
        layer_prefix = ["s", "c", "f"] * self.num_layers
        stand_order = [prefix + '-' + str(l_idx) for l_idx, prefix in enumerate(layer_prefix)]
        return [TransformerDecoderSublayer(args, layer_arch, no_encoder_attn) for layer_arch in stand_order], stand_order

    def reset_space(self):
        self.decoder_current_order = self.vanilla_order
    
    def get_space(self):
        return self.decoder_current_order

    def reorder_search_space(self):
        candidate_order = [["s", "f", "c"], ["c", "s", "f"], ["c", "f", "s"], ["f", "s", "c"], ["f", "c", "s"]]
        block_list = []
        for order in candidate_order:
            block_order = []
            for l_num in range(self.num_layers):
                offset = l_num * 3
                for predfix in order:
                    if predfix == 's':
                        suffix = offset
                    elif predfix == 'c':
                        suffix = offset + 1
                    else:
                        suffix = offset + 2
                    block_order.append(predfix + "-" + str(suffix))
            block_list.append(block_order)
        return block_list
    
    def reorder_current_order(self):
        order_index = np.random.randint(0, 5)
        self.decoder_current_order = self.reorder_space[order_index]
    
    def reorder_by_probs(self, p):
        reorder_oreder = []
        dropout_probs = torch.empty(self.num_layers).uniform_()
        for sublayer in range(self.num_layers):
            start = sublayer * 3
            if dropout_probs[sublayer] < p:
                reorder_oreder.extend(np.random.permutation(self.decoder_current_order[start: start + 3]))
            else:
                reorder_oreder.extend(self.decoder_current_order[start: start + 3])
        self.decoder_current_order = reorder_oreder

    def drop_current_layer(self):
        dropout_oreder = []
        skip_layer_num = np.random.randint(0, 6)
        for sublayer in range(self.num_layers):
            if sublayer == skip_layer_num:continue
            start = sublayer * 3
            dropout_oreder.extend(self.decoder_current_order[start: start + 3])
        self.decoder_current_order = dropout_oreder

    def extract_features_scriptable(
            self,
            prev_output_tokens,
            encoder_out: Optional[Dict[str, List[Tensor]]],
            incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
            full_context_alignment: bool = False,
            alignment_layer: Optional[int] = None,
            alignment_heads: Optional[int] = None,
        ):

        if alignment_layer is None:
            alignment_layer = self.num_layers - 1

        # embed positions
        positions = (
            self.embed_positions(
                prev_output_tokens, incremental_state=incremental_state
            )
            if self.embed_positions is not None
            else None
        )

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]
        
        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(prev_output_tokens)

        if self.quant_noise is not None:
            x = self.quant_noise(x)

        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        if positions is not None:
            x += positions

        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)

        x = self.dropout_module(x)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        self_attn_padding_mask: Optional[Tensor] = None
        if self.cross_self_attention or prev_output_tokens.eq(self.padding_idx).any():
            self_attn_padding_mask = prev_output_tokens.eq(self.padding_idx)

        # decoder layers
        attn: Optional[Tensor] = None
        inner_states: List[Optional[Tensor]] = [x]

        for layer_arch in self.decoder_current_order:
            
            _, layer_suffix = layer_arch.split("-")
            idx = int(layer_suffix)
            layer = self.layers[idx]

            if incremental_state is None and not full_context_alignment:
                self_attn_mask = self.buffered_future_mask(x)
            else:
                self_attn_mask = None

            x, layer_attn, _ = layer(
                x,
                encoder_out["encoder_out"][0]
                if (encoder_out is not None and len(encoder_out["encoder_out"]) > 0)
                else None,
                encoder_out["encoder_padding_mask"][0]
                if (
                    encoder_out is not None
                    and len(encoder_out["encoder_padding_mask"]) > 0
                )
                else None,
                incremental_state,
                self_attn_mask=self_attn_mask,
                self_attn_padding_mask=self_attn_padding_mask,
                need_attn=bool((idx == alignment_layer)),
                need_head_weights=bool((idx == alignment_layer)),
            )
            inner_states.append(x)
            if layer_attn is not None and idx == alignment_layer:
                attn = layer_attn.float().to(x)

        if attn is not None:
            if alignment_heads is not None:
                attn = attn[:alignment_heads]

            # average probabilities over heads
            attn = attn.mean(dim=0)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        return x, {"attn": [attn], "inner_states": inner_states}


@register_model_architecture("variants_transformer", "variants_transformer")
def transformer_iwslt_de_en(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 1024)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 512)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 1024)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    base_architecture(args)

@register_model_architecture("variants_transformer", "variants_transformer_big")
def transformer_vaswani_wmt_en_de_big(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1024)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 4096)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 16)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 1024)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 4096)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 16)
    args.dropout = getattr(args, "dropout", 0.3)
    base_architecture(args)