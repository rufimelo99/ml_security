from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from ml_security.attacks.membership_inference_attack import create_attack_dataloader
from ml_security.logger import logger
from ml_security.utils.utils import get_device

DEVICE = get_device(allow_mps=True)


class MSMarcoDataset(torch.utils.data.Dataset):
    def __init__(self, passages: List[str]):
        self.passages = passages

    def __len__(self):
        return len(self.passages)

    def __getitem__(self, idx):
        return self.passages[idx]


# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[
        0
    ]  # First element of model_output contains all token embeddings
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


def create_dataset_from_ms_marco(ms_marco, split="train"):
    passages = []
    for i in range(len(ms_marco[split])):
        for j in ms_marco[split][i]["passages"]["passage_text"]:
            passages.append(j)
    return passages


# Load model from HuggingFace Hub
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-mpnet-base-v2")
ms_marco = load_dataset("microsoft/ms_marco", "v1.1")


train_passages = create_dataset_from_ms_marco(ms_marco, split="train")
holdout_passages = create_dataset_from_ms_marco(ms_marco, split="validation")


train_dataloader = DataLoader(
    MSMarcoDataset(train_passages), batch_size=32, shuffle=True
)
holdout_dataloader = DataLoader(
    MSMarcoDataset(holdout_passages), batch_size=32, shuffle=True
)


def get_confidence_scores(model, dataloader, device):
    confidence_scores = []
    for batch in tqdm(dataloader):
        # Tokenize sentences
        encoded_input = tokenizer(
            batch, padding=True, truncation=True, return_tensors="pt"
        )

        # Compute token embeddings
        with torch.no_grad():
            model_output = model(**encoded_input)

        # Perform pooling
        sentence_embeddings = mean_pooling(
            model_output, encoded_input["attention_mask"]
        )

        # Normalize embeddings
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

        confidence_scores.append(sentence_embeddings)

    return torch.cat(confidence_scores)


attack_dataloader, attack_labels = create_attack_dataloader(
    train_dataloader,
    holdout_dataloader,
    model,
    device="cpu",
    get_confidence_scores=get_confidence_scores,
)


class AttackerModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=1):
        super(AttackerModel, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x


attack_model = AttackerModel(input_dim=768, hidden_dim=100)
criterion = nn.BCELoss()
optimizer = optim.Adam(attack_model.parameters(), lr=0.001)

# Trains the attack model.
attack_model.train()

for epoch in range(1):
    for data, target in tqdm(attack_dataloader):
        data, target = data.to(DEVICE), target.to(DEVICE)
        optimizer.zero_grad()
        output = attack_model(data)
        loss = criterion(output, target.unsqueeze(1))
        loss.backward()
        optimizer.step()

    if epoch % 10 == 0:
        logger.info("Finished epoch", epoch=epoch, loss=loss.item())

attack_model.eval()

attack_predictions = []
with torch.no_grad():
    for data, target in tqdm(attack_dataloader):
        data, target = data.to(DEVICE), target.to(DEVICE)
        output = attack_model(data)
        attack_predictions.append(output.cpu().numpy())

attack_predictions = np.concatenate(attack_predictions)

# Calculate the accuracy of the attack model.
attack_accuracy = np.mean((attack_predictions > 0.5) == attack_labels)
logger.info("Attack stats", accuracy=attack_accuracy)
