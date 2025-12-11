import torch
from utils import load_config_file, get_device
from tqdm import tqdm
import open_clip
import torch.nn as nn
import numpy as np

device = get_device()

# TODO: separate config.yaml into a train options and test options
def evaluate(model, test_loader, top_ks=[1]):
    model.to(device)
    model.eval()

    correct_at_k = {k: 0 for k in top_ks}
    total = 0

    with torch.no_grad():
        for images, ids, _ in tqdm(test_loader):
            images = images.to(device)
            labels = ids.to(device)
            probs = model(images) 

            topk_preds = torch.topk(probs, k=max(top_ks), dim=1).indices

            for k in top_ks:
                correct = topk_preds[:, :k].eq(labels.view(-1, 1)).any(dim=1).sum().item()
                correct_at_k[k] += correct

            total += labels.size(0)

    topk_accuracies = [correct_at_k[k] / total for k in top_ks]
    return topk_accuracies