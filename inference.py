import torch
import torch.nn as nn

import torch.nn.functional as F
from gemma import PaliGemma, KVCache

def top_p(probs, p=0.7):
    # (batch_size, vocab_size)
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    # prefix sum
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, 1)  # returns an index
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token

def sample(
    model_inputs, # Dict
    model: PaliGemma,
    prompt: str,
    max_tokens_to_generate: int,
    temperature: float,
    top_p: float,
    do_sample: bool,
    stop_token
):
    input_ids = model_inputs["input_ids"]
    attention_mask = model_inputs["attention_mask"]
    pixel_values = model_inputs["pixel_values"]

    kv_cache = KVCache()

    generated_tokens = []
    
    for _ in range(max_tokens_to_generate):
        outputs = model(
            image_features=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            kv_cache=kv_cache
        )

        kv_cache = outputs["kv_cache"]
        next_token_logits = outputs["logits"][:, -1, :]

        if do_sample:
            next_token_logits = torch.softmax(next_token_logits / temperature, dim=-1)
            next_token = top_p(next_token_logits, p=top_p)
        else:
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        next_token = next_token.squeeze(0)
        generated_tokens.append(next_token)
        if next_token == stop_token:
            break
        input_ids = next_token.unsqueeze(-1)
        attention_mask = torch.cat(
            [attention_mask, torch.ones((1, 1), device=input_ids.device)], dim=-1
        )

    generated_tokens = torch.cat(generated_tokens, dim=-1)
