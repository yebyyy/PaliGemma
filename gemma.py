import torch
import torch.nn as nn
from siglip import SiglipVisionConfig, SigLipVisionModel

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
        