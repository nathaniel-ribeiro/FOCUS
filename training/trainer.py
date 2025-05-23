import torch
import torch.optim as optim
import torch.nn as nn
from utils import load_config_file
import itertools
from tqdm import tqdm
import open_clip
import math
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

class Trainer:
    def __init__(self, options, train_loader, val_loader, test_loader, labels):
        self.device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
        self.options = options
       
        self.LEARNING_RATE = self.options.learning_rate
        self.INITIAL_TEMP = self.options.initial_temp
        self.EPOCHS = self.options.epochs
        self.SAVE_FREQ = self.options.save_freq
        
        self.model = CLIPClassifier(options.model_name, options.pretrained, labels).to(self.device)
        
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
    
    def train(self):    
        self.model.train()

        for param in self.model.parameters():
            param.requires_grad = True

        optimizer = optim.AdamW(self.model.parameters(), lr=self.LEARNING_RATE)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(self.EPOCHS):
            self.model.train()
            loss_for_epoch = 0.0

            for batch_idx, (images, _, prompts) in tqdm(enumerate(self.train_loader)):
                optimizer.zero_grad()
                images = images.to(self.device)
                tokenized = self.model.tokenize(prompts).to(self.device)

                # omitting preprocessing for images because Albumentations pipeline handles this
                image_features = self.model.encode_image(images).to(self.device)
                text_features = self.model.encode_text(tokenized).to(self.device)
                
                temperature = torch.clamp(self.model.temperature.exp(), max=100.0)
                
                logits_per_image = temperature * image_features @ text_features.T
                logits_per_text = logits_per_image.T

                targets = torch.arange(images.size(0), dtype=torch.long).to(self.device)

                loss_i2t = criterion(logits_per_image, targets)
                loss_t2i = criterion(logits_per_text, targets)
                loss = (loss_i2t + loss_t2i) / 2.0
                loss_for_epoch += loss.item()

                loss.backward()
                optimizer.step()
                
            avg_loss_for_epoch = loss_for_epoch / len(self.train_loader)
            print(f"Training loss for epoch {epoch}: {avg_loss_for_epoch:.4f}")
    
    def validate(self):
        pass
    
    def test():
        pass
