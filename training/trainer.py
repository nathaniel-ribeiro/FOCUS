import torch
import torch.optim as optim
import torch.nn as nn
from utils import load_config_file, get_device
import itertools
from tqdm import tqdm
import open_clip
import math
import numpy as np
from classifier import CLIPClassifier
from evaluator import evaluate
from copy import deepcopy

device = get_device()

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
        for param in self.model.parameters():
            param.requires_grad = True

        optimizer = optim.AdamW(self.model.parameters(), lr=self.LEARNING_RATE)
        criterion = nn.CrossEntropyLoss()
        best_val_loss = np.inf
        best_model = deepcopy(self.model)
        for epoch in range(self.EPOCHS):
            self.model.train()
            train_loss = 0.0
            for images, _, prompts in self.train_loader:
                optimizer.zero_grad()
                images = images.to(device)
                logits_per_image, logits_per_text = self.model.forward_contrastive(images, prompts)

                targets = torch.arange(images.size(0), dtype=torch.long).to(device)

                loss_i2t = criterion(logits_per_image, targets)
                loss_t2i = criterion(logits_per_text, targets)
                loss = (loss_i2t + loss_t2i) / 2.0
                train_loss += loss.item()

                loss.backward()
                optimizer.step()
            
            avg_train_loss = train_loss / len(self.train_loader)
            
            self.model.eval()
            val_loss = 0.0
            for images, _, prompts in self.val_loader:
                images = images.to(device)
                logits_per_image, logits_per_text = self.model.forward_contrastive(images, prompts)

                targets = torch.arange(images.size(0), dtype=torch.long).to(device)

                loss_i2t = criterion(logits_per_image, targets)
                loss_t2i = criterion(logits_per_text, targets)
                loss = (loss_i2t + loss_t2i) / 2.0
                val_loss += loss.item()
            
            avg_val_loss = val_loss / len(self.val_loader)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model = deepcopy(self.model)

            print(f"Train loss: {avg_train_loss} \t Val loss: {avg_val_loss}")
        self.model = deepcopy(best_model)
            
