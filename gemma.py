import torch
import torch.nn as nn
from siglip import SiglipVisionConfig, SigLipVisionModel

class KVCache():

    def __init__(self):
        self.key_cache = []  # each layer has a key cache, the index is the layer index
        self.value_cache = []

    def num_items(self):
        if len(self.key_cache) == 0:
            return 0
        else:
            # (batch_size, num_kv_heads, seq_len, head_dim)
            return self.key_cache[0].shape[-2]
    
    def update(self, key, value, layer_idx):
        # add the current key and value to the cache
        if len(self.key_cache) <= layer_idx:
            # since index of the key_cache is the layer index, we need to
            # create the kv_cache
            self.key_cache.append(key)
            self.value_cache.append(value)
        else:
            # if the cache for this layer already exists, update it
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key], dim=-2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value], dim=-2)

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

class GemmaConfig():

    def __init__(
        self,
        vocab_size,
        hidden_size,
        intermediate_size,
        num_hidden_layers,
        num_attention_heads,
        num_key_value_heads,
        head_dim=256,
        max_position_embeddings=8192,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        attention_bias=False,
        attention_dropout=0.0,
        pad_token_id=None,
        **kwargs,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim
        self.num_key_value_heads = num_key_value_heads
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.pad_token_id = pad_token_id

class PaliGemmaConfig():

    def __init__(
        self,
        vision_config=None,
        text_config=None,
        ignore_index=-100,   
        image_token_index=256000,
        vocab_size=257152,
        projection_dim=2048,     # output dimension of the image embedding
        hidden_size=2048,
        pad_token_id=None,
        **kwargs,
    ):
        super().__init__()
        self.ignore_index = ignore_index
        self.image_token_index = image_token_index
        self.vocab_size = vocab_size
        self.projection_dim = projection_dim
        self.hidden_size = hidden_size
        self.vision_config = vision_config
        self.is_encoder_decoder = False
        self.pad_token_id = pad_token_id

        self.vision_config = SiglipVisionConfig(**vision_config)
        self.text_config = text_config

        self.text_config = GemmaConfig(**text_config, pad_token_id=pad_token_id)
        self.vocab_size = self.text_config.vocab_size

        self.text_config.num_image_tokens = (self.vision_config.image_size // self.vision_config.patch_size) ** 2
        self.vision_config.projection_dim = projection_dim

class PaliGemmaLinearProjector(nn.Module):

    def __init__(self, config: PaliGemmaConfig):
        super().__init__()
        self.config = config
        self.projection = nn.Linear(config.hidden_size, config.projection_dim)

    def forward(self, image_embedding):
        image_embedding = self.projection(image_embedding)
        return image_embedding

class GemmaMLP(nn.Module):

    def __init__(self, config: GemmaConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size)

    def forward(self, x):
        y = self.gate_proj(x)
        y = nn.functional.gelu(y, approximate="tanh")
        j = self.up_proj(x)
        z = y * j
        z = self.down_proj(z)
        # self.down_proj(nn.functional.gelu(self.gate_proj(x), approximate="tanh") * self.up_proj(x))
        return z

def repeat_kv(kv, num_kv_groups):
    batch_size, num_kv_heads, seq_len, head_dim = kv.shape
    if num_kv_groups == 1:
        return kv
    kv = kv[:, :, None, :, :].expand(batch_size, num_kv_heads, num_kv_groups, seq_len, head_dim) # repeats num_kv_groups times of the last two dimensions
    return kv.reshape(batch_size, num_kv_heads * num_kv_groups, seq_len, head_dim)

class GemmaRotaryEmbedding(nn.Module):

    def __init__(self, dim, max_position_embeddings, base=10000, device=None):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)  # persistent means not included in state_dict
    
    @torch.no_grad()
    def forward(self, x, position_ids, seq_len=None):
        self.inv_freq = self.inv_freq.to(x.device)
        inv_freq_expand = self.inv_freq[None, :, None].expand(position_ids[0], -1, -1)  # (batch, dim//2, 1)
        position_ids_expand = position_ids[:, None, :]
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expand.float() @ position_ids_expand.float()).transpose(1, 2)
            # emb: [Batch_Size, Seq_Len, Head_Dim]
            emb = torch.cat((freqs, freqs), dim=-1)
            # cos, sin: [Batch_Size, Seq_Len, Head_Dim]
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def rotate_half(x):
    # Build the [-x2, x1, -x4, x3, ...] tensor for the sin part of the positional encoding.
    x1 = x[..., : x.shape[-1] // 2] # Takes the first half of the last dimension
    x2 = x[..., x.shape[-1] // 2 :] # Takes the second half of the last dimension
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim) # Add the head dimension
    sin = sin.unsqueeze(unsqueeze_dim) # Add the head dimension
    # Apply the formula (34) of the Rotary Positional Encoding paper.
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class GemmaAttention(nn.Module):

    def __init__(self, config: GemmaConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.hidden_size = config.hidden_size
        self.attention_dropout = config.attention_dropout
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.num_kv_heads = config.num_key_value_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads  # how many query heads share a key and value head
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True

        assert self.hidden_size % self.num_heads == 0, "hidden_size must be divisible by num_heads"

        # (batch, num_patches, 1024) -> (batch, num_patches, 8 * 128)
        self.q_proj = nn.Linear(self.hidden_size, self.head_dim * self.num_heads, bias=config.attention_bias)
        # (batch, num_patches, 1024) -> (batch, num_patches, 1 * 128)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias)
        # (batch, num_patches, 8 * 128) -> (batch, num_patches, 1024)
        self.out_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=config.attention_bias)
        self.rotary_emb = GemmaRotaryEmbedding(
            self.head_dim, 
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )

    def forward(self, hidden_states, attention_mask, position_ids, kv_cache, **kwargs):
        batch_size, seq_len, embed_dim = hidden_states.shape
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # rotary position embedding
        cos, sin = self.rotary_emb(position_ids, self.layer_idx)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # use the kv_cache instead of k and v to calculate the attention
        if kv_cache is not None:
            # update k, v to all the keys and values from the cache
            k, v = kv_cache.update(k, v, self.layer_idx)

        # repeat to match the number of heads
        # every num_kv_groups query heads share the same key and value head
        # (no need if we are using CUDA)
        k = repeat_kv(k, self.num_kv_groups)
        v = repeat_kv(v, self.num_kv_groups)

        # (batch, num_heads, seq_len_q, seq_len_kv)
        attention_weights = torch.matmul(q, k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        
        assert attention_mask is not None
        attention_weights = attention_weights + attention_mask

        # softmax along the last dimension
        attention_weights = nn.functional.softmax(attention_weights, dim=-1)
        attention_weights = nn.functional.dropout(attention_weights, p=self.attention_dropout, training=self.training)

        # (batch, num_heads, seq_len_q, head_dim)
        attention_output = torch.matmul(attention_weights, v)

        # (batch, seq_len_q, num_heads, head_dim)
        attention_output = attention_output.transpose(1, 2).contiguous()
        attention_output = attention_output.reshape(batch_size, seq_len, self.hidden_size)
        
        attention_output = self.out_proj(attention_output)
        return attention_output, attention_weights


class GemmaDecoderLayer(nn.Module):

    def __init__(self, config: GemmaConfig, layer_id):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attention = GemmaAttention(config, layer_id)
        self.mlp = GemmaMLP(config)
        self.input_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, attention_mask, hidden_states, position_ids, kv_cache):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attention(hidden_states=hidden_states, attention_mask=attention_mask, position_ids=position_ids, kv_cache=kv_cache)
        hidden_states += residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states += residual
        return hidden_states

class GemmaRMSNorm(nn.Module):

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))  # one parameter for each feature

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keep_dim=True) + self.eps)  # keep_dim=True means keep the last dimension
    
    def forward(self, x):
        output = self._norm(x)
        output = output * self.weight

class Gemma(nn.Module):

    def __init__(self, config: GemmaConfig):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=self.padding_idx)
        self.layers = nn.ModuleList(
            [GemmaDecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def get_input_embeddings(self):
        return self.embed_tokens
    
    def forward(self, attention_mask, position_ids, input_embeddings, kv_cache):
        hidden_states = input_embeddings
        normalizer = torch.tensor(self.config.hidden_size ** 0.5, dtype=hidden_states.dtype)
        hidden_states = hidden_states * normalizer
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask, position_ids, kv_cache)
        hidden_states = self.norm(hidden_states)
        return hidden_states


class GemmaForCausalLM(nn.Module):
    # ForCausalLM is the LM + linear layer

    def __init__(self, config: GemmaConfig):
        super().__init__()
        self.config = config
        self.model = Gemma(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def get_input_embeddings(self):
        return self.model.embed_tokens
    
    def tie_weights(self):
        self.lm_head.weight = self.model.embed_tokens.weight

    def forward(self, attention_mask, position_ids, input_embeddings, kv_cache):
        outputs = self.model(attention_mask=attention_mask, position_ids=position_ids, input_embeddings=input_embeddings, kv_cache=kv_cache)
        logits = self.lm_head(outputs)
        return_data = {
            "logits": logits,
        }
        if kv_cache is not None:
            return_data["kv_cache"] = kv_cache
        return return_data

class PaliGemma(nn.Module):
    
    def __init__(self, config: PaliGemmaConfig):
        super().__init__()
        self.config = config
        self.vision_tower = SigLipVisionModel(config.vision_config)
        self.linear_projector = PaliGemmaLinearProjector(config)
        self.vocab_size = config.vocab_size

        language_model = GemmaForCausalLM(config.text_config)
        self.language_model = language_model

        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1

    def tie_weights(self):
        return self.language_model.tie_weights()
    
    def merge_embeddings(self, image_feature, input_embeddings, input_ids, attention_mask, kv_cache):
        _, _, embed_dim = image_feature.shape
        batch_size, seq_len = input_ids.shape
        dtype, device = input_embeddings.dtype, input_embeddings.device

        scaled_image_feature = image_feature * (self.config.hidden_size ** -0.5)

        combined_embeddings = torch.zeros(batch_size, seq_len, embed_dim, dtype=dtype, device=device)
        # [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
        text_mask = (input_ids != self.config.image_token_index) & (input_ids != self.pad_token_id)
        # [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
        image_mask = (input_ids == self.config.image_token_index)
        # we don't have any padding
        pad_mask = (input_ids == self.pad_token_id)

        text_mask = text_mask.unsqueeze(-1).expand(-1, -1, embed_dim)  # expand(-1,-1,embed_dim) means expand the last dimension
        image_mask = image_mask.unsqueeze(-1).expand(-1, -1, embed_dim)
        pad_mask = pad_mask.unsqueeze(-1).expand(-1, -1, embed_dim)

        # if text, then copy input embedding, else copy zeros (input_embeddings has junk image embed + good text embed)
        combined_embeddings = torch.where(text_mask, input_embeddings, combined_embeddings)
        # same thing as torch.where, but image_feature has different shape then combined_embeddings (missing text embeddings!)
        combined_embeddings = combined_embeddings.masked_scatter(image_mask, scaled_image_feature)
        combined_embeddings = torch.where(pad_mask, torch.zeros_like(combined_embeddings), combined_embeddings)

        q_len = input_embeddings.shape[1]

        # attention mask does not mask out anything
        # softmax(-inf) = 0, softmax(0) = 1, using 0 represents not masking out
        if kv_cache is None or kv_cache.num_itmes() == 0:
            # kv_len set to q_len to prefill the cache
            causal_mask = torch.full((batch_size, q_len, q_len), fill_value=0, dtype=dtype, device=device)
        else:
            assert q_len == 1  # always the last query token
            kv_len = kv_cache.num_items() + q_len  # add 1 for the new token
            causal_mask = torch.full((batch_size, q_len, kv_len), fill_value=0, dtype=dtype, device=device)

        # (batch, q_len, kv_len) -> (batch, num_heads_q, q_len, kv_len)
        causal_mask = causal_mask.unsqueeze(-1)

        if kv_cache is not None and kv_cache.num_items() > 0:
            position_ids = attention_mask.cumsum(-1)[:, -1]
            if position_ids.dim() == 1:
                position_ids = position_ids.unsqueeze(0)
        else:
            position_ids = (attention_mask.cumsum(-1)).masked_fill_((attention_mask == 0),  1).to(device)

        return combined_embeddings, causal_mask, position_ids

    def forward(
        self,
        input_ids, # input_ids are the token embeddings with the image embedding placeholders
        pixel_values,  # image to feed to siglip to get embeddings
        attention_mask,
        kv_cache
    ):
        assert torch.all(attention_mask == 1), "input should not be padded"
        
        # get the embeddings for the text
        input_embeddings = self.language_model.get_input_embeddings()(input_ids)

        # get the image embeddings
        image_feature = self.vision_tower(pixel_values.to(input_embeddings.dtype))
        image_feature = self.linear_projector(image_feature)

        # merge image and text embeddings
        input_embeddings, attention_mask, position_ids = self.merge_embeddings(image_feature, input_embeddings, input_ids, attention_mask, kv_cache)

        output = self.language_model(
            attention_mask=attention_mask,
            image_feature=pixel_values,
            input_ids=input_ids,
            kv_cache=kv_cache,
        )
        
        return output
