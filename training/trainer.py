import torch
import torch.optim as optim
import torch.nn as nn
from utils import load_config_file
import itertools
from tqdm import tqdm
import open_clip
import math
import numpy as np
from classifier import CLIPClassifier        

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
                logits_per_image, logits_per_text = self.model.forward_contrastive(images, prompts)

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
