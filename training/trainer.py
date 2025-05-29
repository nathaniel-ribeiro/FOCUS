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
    def __init__(self, model, options, train_loader, val_loader, test_loader):
        self.options = options
       
        self.LEARNING_RATE = self.options.learning_rate
        self.EPOCHS = self.options.epochs
        self.SAVE_FREQ = self.options.save_freq
        
        self.model = model
        
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
    
    def train(self):    
        self.model.train()

        for param in self.model.parameters():
            param.requires_grad = True

        optimizer = optim.AdamW(self.model.parameters(), lr=self.LEARNING_RATE)
        criterion = nn.CrossEntropyLoss()

        for epoch in tqdm(range(self.EPOCHS)):
            self.model.train()
            loss_for_epoch = 0.0

            for images, _, prompts in self.train_loader:
                optimizer.zero_grad()
                images = images.to(self.device)
                logits_per_image, logits_per_text = self.model.forward_contrastive(images, prompts)

                targets = torch.arange(images.size(0), dtype=torch.long).to(self.model.device)

                loss_i2t = criterion(logits_per_image, targets)
                loss_t2i = criterion(logits_per_text, targets)
                loss = (loss_i2t + loss_t2i) / 2.0
                loss_for_epoch += loss.item()

                loss.backward()
                optimizer.step()
                
            avg_loss_for_epoch = loss_for_epoch / len(self.train_loader)
