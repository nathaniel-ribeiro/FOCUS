# from inat_data import make_dataloaders, get_labels
from micronet_data import make_dataloaders, get_labels
from utils import load_config_file, get_device
from trainer import Trainer, CLIPClassifier
from evaluator import evaluate
import torch

device = get_device()
options = load_config_file('config.yaml')
train_loader, val_loader, test_loader = make_dataloaders()
labels = get_labels()

print(f"Memory Allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
print(f"Memory Reserved: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")

model = CLIPClassifier(options.model_name, options.pretrained, labels).to(device)

print(f"Memory Allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
print(f"Memory Reserved: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
# trainer = Trainer(model, options, train_loader, val_loader, test_loader)
# trainer.train()
top_ks = [1, 3, 5]
top_k_accuracies = evaluate(model, test_loader, top_ks = top_ks)

for k, acc in zip(top_ks, top_k_accuracies):
    print(f"Top-{k} acc: {(acc * 100.0):.2f}%")