import types

import torch
from torch.nn import functional as F


def _forward(self, hidden_states, attention_mask=None, head_mask=None,
             encoder_hidden_states=None, encoder_attention_mask=None,
             past_key_value=None, output_attentions=False):
    if (encoder_hidden_states is not None or past_key_value is not None or
            output_attentions or head_mask is not None or self.position_embedding_type != "rotary"):
        return self._original_full_attention(
            hidden_states, attention_mask, head_mask, encoder_hidden_states,
            encoder_attention_mask, past_key_value, output_attentions)
    query = self.transpose_for_scores(self.query(hidden_states))
    key = self.transpose_for_scores(self.key(hidden_states))
    value = self.transpose_for_scores(self.value(hidden_states))
    query = query * self.attention_head_size ** -0.5
    query, key = self.rotary_embeddings(query, key)
    contexts = []
    for start in range(0, query.shape[2], self._attention_query_chunk):
        end = min(start + self._attention_query_chunk, query.shape[2])
        scores = torch.matmul(query[:, :, start:end, :], key.transpose(-1, -2))
        if attention_mask is not None:
            mask = attention_mask if attention_mask.shape[-2] == 1 else attention_mask[..., start:end, :]
            scores = scores + mask
        probabilities = F.softmax(scores, dim=-1)
        probabilities = self.dropout(probabilities)
        contexts.append(torch.matmul(probabilities.to(value.dtype), value))
    context = torch.cat(contexts, dim=2).transpose(1, 2).contiguous()
    context = context.view(context.size()[:-2] + (self.all_head_size,))
    return (context,)


def enable_query_chunked_attention(model, query_chunk=128):
    if query_chunk <= 0:
        raise ValueError("query_chunk must be positive")
    count = 0
    for module in model.modules():
        if module.__class__.__name__ != "RiNALMoSelfAttention":
            continue
        if module.position_embedding_type != "rotary" or module.is_decoder:
            raise ValueError("query chunking supports only the full bidirectional rotary encoder")
        module._original_full_attention = module.forward
        module._attention_query_chunk = query_chunk
        module.forward = types.MethodType(_forward, module)
        count += 1
    if count != 33:
        raise AssertionError(f"expected 33 RiNALMo attention modules, got {count}")
    model._attention_query_chunk = query_chunk
    return count
