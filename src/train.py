import os
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm

from src.loading import BachDataset
from src.model import Autoencoder


def train_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Autoencoder().to(device)
    dataset = BachDataset("bach-dataset")
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
    optimizer = optim.Adam(model.parameters())
    criterion = nn.MSELoss()
    n_epochs = 100

    train_loop(model, criterion, optimizer, dataloader, n_epochs=n_epochs)


def train_loop(
    model: nn.Module,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    dataloader: torch.utils.data.DataLoader,
    n_epochs: int = 100
) -> None:
    for epoch in tqdm(range(n_epochs)):
        for sample in dataloader:
            train_step(model, criterion, optimizer, sample)
        save_model(model, epoch)


def train_step(
    model: nn.Module,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    sample: torch.Tensor,
) -> None:
    sample = sample.to(next(model.parameters()).device)
    reconstruction = model(sample)
    loss = criterion(reconstruction, sample)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def save_model(model: nn.Module, epoch: int) -> None:
    now = datetime.now()
    year = now.year
    month = now.month
    day = now.day
    hour = now.hour
    minute = now.minute
    time = f"{year}-{month}-{day}-{hour}-{minute}"

    models_directory = os.path.join("models", time)
    os.makedirs(models_directory, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(models_directory, f"{epoch}.pt"))


if __name__ == "__main__":
    train_model()
