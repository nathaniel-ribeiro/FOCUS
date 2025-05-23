import torch
from utils import load_config_file
from tqdm import tqdm
import open_clip
import torch.nn as nn
import numpy as np

# TODO: separate config.yaml into a train options and test options
def evaluate(model, test_loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    all_preds = []
    all_ids = []

    with torch.no_grad():
        for images, ids, _ in tqdm(test_loader):
            images = images.to(device)
            predicted_probabilities = model(images).to(device)
            predicted_classes = torch.argmax(predicted_probabilities, dim=1).to(device)

            all_preds.append(predicted_classes.cpu().numpy())
            all_ids.append(ids.numpy())

    all_preds = np.concatenate(all_preds)
    all_ids = np.concatenate(all_ids)

    accuracy = np.mean(all_preds == all_ids)
    print(f"Test Accuracy: {accuracy:.4f}")
    return accuracy, all_preds, all_ids