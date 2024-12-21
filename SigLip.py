from typing import Tuple, Optional
import torch
import torch.nn as nn

class SiglipVisionConfig:

    def __init__(
        self,
        hidden_size=768,                 # embedding size
        intermediate_size=3072,          # feedforward linear layer size
        num_hidden_layers=12,           
        num_attention_heads=12,
        num_channels=3,
        image_size=224,
        patch_size=16,                   # each patch is 16x16 pixels
        layer_norm_eps=1e-6,
        attention_dropout=0.0,
        num_image_tokens: int = None,     # number of patches
        **kwargs
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.image_size = image_size
        self.attention_dropout = attention_dropout
        self.layer_norm_eps = layer_norm_eps
        self.num_image_tokens = num_image_tokens

class SigLipVisionEmbeddings(nn.Module):
    # convolution to embedding + positional embedding
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,    # so that no overlap
            padding=0
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches
        self.positional_embedding = nn.Embedding(num_embeddings=self.num_positions, embedding_dim=self.embed_dim)
        self.register_buffer(
            "position_ids",                                    # register_buffer: tensor that is not updated during backprop
            torch.arange(self.num_positions).expand((1, -1)),   # (1, num_positions)
            persistent=False                                   # not included in state_dict
        )    

    def forward(self, pixel_values):
        batch_size, num_channels, height, width = pixel_values.shape
        patch_embeddings = self.patch_embedding(pixel_values)   # convolution to embedding
        # (batch, embed_dim, num_patches_height, num_patches_width)
        embeddings = patch_embeddings.flatten(2)             # flatten(2) means flatten the last two dimensions
        # (batch, embed_dim, num_patches)                   # flatten to make to a list of embeddings
        embeddings = embeddings.transpose(1, 2)              
        # (batch, num_patches, embed_dim)
        # this would allow us to have a sequence of patches with their embeddings
        embeddings = embeddings + self.positional_embedding(self.position_ids)  # positional embedding of all posible positions
        return embeddings



class SigLipVisionTransformer(nn.Module):
    
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size

        self.embeddings = SigLipVisionEmbeddings(config)  # convolution to embedding + positional embedding
        self.encoder = SiglipVisionEncoder(config)      # encoder in transformer
        self.post_layernorm = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
    
    def forward(self, pixel_values):
        embedding = self.embeddings(pixel_values)
        encoder_output = self.encoder(embedding)
        post_layernorm = self.post_layernorm(encoder_output)
        return post_layernorm

class SigLipVisionModel(nn.Module):
    
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.vision_model = SigLipVisionTransformer(config)
    def forward(self, pixel_values):
        return self.vision_model(pixel_values)
        # (batch, channel, height, width) -> (batch, num_patches, embed_dim)