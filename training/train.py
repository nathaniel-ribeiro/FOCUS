# from inat_data import make_dataloaders, get_labels
from micronet_data import make_dataloaders, get_labels
from utils import load_config_file, get_device
from trainer import Trainer, CLIPClassifier
from evaluator import evaluate
import torch
import numpy as np
from torchvision.transforms.functional import to_pil_image

device = get_device()
options = load_config_file('config.yaml')
train_loader, val_loader, test_loader = make_dataloaders()
labels = get_labels()

model = CLIPClassifier(options.model_name, options.pretrained, labels).to(device)

std = torch.tensor(np.array([0.26862954, 0.26130258, 0.27577711])).view(3, 1, 1)
mean = torch.tensor(np.array([0.48145466, 0.4578275, 0.40821073])).view(3, 1, 1)

for images, ids, prompts in train_loader:
    img = images.squeeze()[0]
    img = img * std + mean
    pil = to_pil_image(img.cpu())

    out_path = "../data/example.jpg"
    pil.save(out_path)

    id = ids.squeeze()[0]
    species = labels[id]
    prompt = prompts[0]

    assert species in prompt, "The ground truth species and prompt did not match. Prompt generation could be buggy or the dataset could be misaligned."
    break

# trainer = Trainer(model, options, train_loader, val_loader, test_loader)
# trainer.train()

# top_ks = [1, 3, 5]
# top_k_accuracies = evaluate(model, test_loader, top_ks = top_ks)

# for k, acc in zip(top_ks, top_k_accuracies):
#     print(f"Top-{k} acc: {(acc * 100.0):.2f}%")