import torch
import torch.nn as nn
from siglip import SiglipVisionConfig, SigLipVisionModel

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

        language_model = GemmaCausalLM(config.text_config)
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
            position_ids=position_ids,
            input_embeddings=input_embeddings,
            kv_cache=kv_cache,
        )
        
        return output
