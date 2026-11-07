import math
import types

import torch
from transformers import BertModel, BertConfig

from rnajepa.ripe import RipeConfig, RipePositionalBias


def _embedding_without_position(self, input_ids=None, token_type_ids=None, position_ids=None,
                                inputs_embeds=None, past_key_values_length=0):
    if input_ids is not None:
        input_shape = input_ids.size()
    else:
        input_shape = inputs_embeds.size()[:-1]
    if token_type_ids is None:
        token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)
    if inputs_embeds is None:
        inputs_embeds = self.word_embeddings(input_ids)
    token_type_embeddings = self.token_type_embeddings(token_type_ids)
    embeddings = inputs_embeds + token_type_embeddings
    embeddings = self.LayerNorm(embeddings)
    return self.dropout(embeddings)


def _attention_with_ripe(self, hidden_states, attention_mask=None, head_mask=None,
                         encoder_hidden_states=None, encoder_attention_mask=None,
                         past_key_value=None, output_attentions=False):
    mixed_query_layer = self.query(hidden_states)
    key_layer = self.transpose_for_scores(self.key(hidden_states))
    value_layer = self.transpose_for_scores(self.value(hidden_states))
    query_layer = self.transpose_for_scores(mixed_query_layer)
    if past_key_value is not None and len(past_key_value) > 0:
        key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
        value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
    attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    positional_bias = self.ripe(self._ripe_input_ids).to(dtype=attention_scores.dtype)
    attention_scores = attention_scores + positional_bias
    attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    if attention_mask is not None:
        attention_scores = attention_scores + attention_mask
    self._ripe_attention_score_dtype = attention_scores.dtype
    attention_probs = torch.nn.functional.softmax(attention_scores, dim=-1)
    self._ripe_softmax_dtype = attention_probs.dtype
    attention_probs = self.dropout(attention_probs)
    if head_mask is not None:
        attention_probs = attention_probs * head_mask
    context_layer = torch.matmul(attention_probs, value_layer)
    context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
    context_layer = context_layer.view(*new_context_layer_shape)
    return (context_layer, attention_probs) if output_attentions else (context_layer,)


def _forward_ripe(self, *args, **kwargs):
    input_ids = kwargs.get("input_ids", args[0] if args else None)
    if kwargs.get("token_type_ids") is None and input_ids is not None:
        kwargs["token_type_ids"] = torch.zeros_like(input_ids)
    for layer in self.encoder.layer:
        layer.attention.self._ripe_input_ids = input_ids
    return self._original_forward(*args, **kwargs)


def enable_ripe_structrfm(model, mode, vocab=None):
    if mode not in {"log", "r1", "r2", "r3", "r5", "r6"}:
        raise ValueError(mode)
    token_id_map = ()
    if mode in {"r3", "r6"}:
        if vocab is None:
            raise ValueError("explicit nucleotide vocab is required for RIPE content bias")
        token_id_map = [7] * model.config.vocab_size
        for base, index in (("A", 0), ("C", 1), ("G", 2), ("U", 3), ("T", 3)):
            token_id = vocab[base]
            if not 0 <= token_id < len(token_id_map):
                raise ValueError(f"token {base} has out-of-range id {token_id}")
            token_id_map[token_id] = index
        token_id_map = tuple(token_id_map)
    model._original_forward = model.forward
    model.forward = types.MethodType(_forward_ripe, model)
    model.embeddings.forward = types.MethodType(_embedding_without_position, model.embeddings)
    for layer in model.encoder.layer:
        attention = layer.attention.self
        attention.ripe = RipePositionalBias(RipeConfig(
            n_heads=model.config.num_attention_heads, mode=mode,
            token_id_map=token_id_map)).to(
                device=attention.query.weight.device, dtype=attention.query.weight.dtype)
        attention.forward = types.MethodType(_attention_with_ripe, attention)
    model.config.position_embedding_type = "ripe"
    return model


def make_tiny_bert(max_position_embeddings=8):
    config = BertConfig(vocab_size=16, hidden_size=12, num_hidden_layers=1,
                        num_attention_heads=2, intermediate_size=24,
                        max_position_embeddings=max_position_embeddings)
    return BertModel(config)
