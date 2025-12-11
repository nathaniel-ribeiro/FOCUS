import torch
import torch.nn as nn
import open_clip
import numpy as np

class CLIPClassifier(nn.Module):
    def __init__(self, model_name, pretrained, labels):
        super().__init__()
        self.model_name = model_name
        self.pretrained = pretrained
        self.model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained)
        self.tokenizer = open_clip.get_tokenizer(model_name)
        # initial value from Wu et. al, 2018
        self.temperature = nn.Parameter(torch.tensor([np.log(1 / 0.07)], dtype=torch.float32))
        self.labels = labels
        print(len(labels))
    
    def tokenize(self, text):
        return self.tokenizer(text)
    
    def encode_image(self, image):
        return self.model.encode_image(image)
    
    def encode_text(self, tokenized_text):
        return self.model.encode_text(tokenized_text)
    
    def forward(self, image):
        # to device call is necessary bc the tokenizer is not a submodule
        tokenized_labels = self.tokenizer(self.labels).to(image.device)
        text_features = self.model.encode_text(tokenized_labels)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        image_features = self.model.encode_image(image)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        temperature = torch.clamp(self.temperature.exp(), max=100.0)
        logits = temperature * image_features @ text_features.T
        return logits.softmax(dim=-1)
    
    def forward_contrastive(self, images, labels):
        tokenized = self.tokenizer(labels).to(images.device)
        image_features = self.model.encode_image(images)
        text_features = self.model.encode_text(tokenized)
        
        temperature = torch.clamp(self.temperature.exp(), max=100.0)                
        logits_per_image = temperature * image_features @ text_features.T
        logits_per_text = logits_per_image.T
        return logits_per_image, logits_per_text