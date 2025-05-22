import torch
import torch.optim as optim
import torch.nn as nn
from utils import load_config_file
import itertools
from tqdm import tqdm
import open_clip
import math

device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
options = load_config_file('config.yaml')

MODEL_NAME = options.model_name
PRETRAINED = options.pretrained
LEARNING_RATE = options.learning_rate
INITIAL_TEMP = options.initial_temp
EPOCHS = options.epochs
SAVE_FREQ = options.save_freq

model, _, preprocess = open_clip.create_model_and_transforms(
    MODEL_NAME, pretrained=PRETRAINED
)
tokenizer = open_clip.get_tokenizer(MODEL_NAME)
model.to(device)
model.train()

for param in model.parameters():
    param.requires_grad = True

def _build_prompt(taxonomy):
    return f"A photo of a {taxonomy.name}."

temperature = torch.log(torch.tensor([1.0 / INITIAL_TEMP])).to(device)
# TODO: optimize temperature
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()

def train(train_loader):
    global temperature
    
    for epoch in range(EPOCHS):
        model.train()
        loss_for_epoch = 0.0
        
        for batch_idx, (images, _, taxonomies) in tqdm(enumerate(train_loader)):
            optimizer.zero_grad()
            images = images.to(device)
            # omitting preprocessing because Albumentations pipeline handles this
            prompts = [_build_prompt(taxonomy) for taxonomy in taxonomies]
            tokenized = tokenizer(prompts).to(device)
            
            image_features = model.encode_image(images).to(device)
            text_features = model.encode_text(tokenized).to(device)
            
            logits_per_image = temperature.exp() * image_features @ text_features.T
            logits_per_text = logits_per_image.T
            
            targets = torch.arange(images.size(0), dtype=torch.long).to(device)
            
            loss_i2t = criterion(logits_per_image, targets)
            loss_t2i = criterion(logits_per_text, targets)
            loss = (loss_i2t + loss_t2i) / 2.0
            loss_for_epoch += loss.item()
            
            loss.backward()
            optimizer.step()
        
        if epoch % SAVE_FREQ:
            print(f"Average loss for epoch {epoch}: {loss_for_epoch / len(train_loader):.4f}")
            print("TODO: save model here")
            
        