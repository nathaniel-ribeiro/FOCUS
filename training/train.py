from inat_data import make_dataloaders, get_labels
from utils import load_config_file
from trainer import Trainer, CLIPClassifier
from evaluator import evaluate
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

options = load_config_file('config.yaml')
train_loader, test_loader = make_dataloaders()
labels = get_labels()
trainer = Trainer(options, train_loader, None, test_loader, labels)
trainer.train()

# model = CLIPClassifier(options.model_name, options.pretrained, labels).to(device)
# top_ks = [1, 3, 5]
# top_k_accuracies = evaluate(model, test_loader, top_ks = top_ks)