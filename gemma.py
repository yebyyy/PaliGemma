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

    def tie_weights(self):
        return self.language_model.tie_weights()
    
    def forward(
        self,
        input_ids, # input_ids are the token embeddings with the image embedding placeholders
        pixel_values,  # image to feed to siglip to get embeddings
        attention_mask,
        kv_cache
    ):
        assert torch.all(attention_mask == 1), "input should not be padded"
        
        input_embeddings = self.language_model.get_input_embeddings()(input_ids)

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
