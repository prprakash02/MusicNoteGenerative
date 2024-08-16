import os

from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from src.loading import BachDataset
from src.model import Autoencoder



def train_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    writer = SummaryWriter(os.path.join('models', CURRENT_TIME))

    model = Autoencoder().to(device)
    dataset = BachDataset("bach-dataset")
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
    optimizer = optim.Adam(model.parameters())
    criterion = nn.MSELoss()
    n_epochs = 100

    train_loop(model, criterion, optimizer, dataloader, n_epochs=n_epochs, writer=writer)


def train_loop(
    model: nn.Module,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    dataloader: torch.utils.data.DataLoader,
    n_epochs: int = 100,
    writer: SummaryWriter = None,
) -> None:
    for epoch in tqdm(range(n_epochs)):
        losses = []
        for sample in dataloader:
            train_step(model, criterion, optimizer, sample, losses)

        save_model(model, epoch)
        avg_loss = np.mean(losses)
        if writer is not None:
            writer.add_scalar('loss', avg_loss, epoch)


def train_step(
    model: nn.Module,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    sample: torch.Tensor,
    losses: list[float],
) -> None:
    sample = sample.to(next(model.parameters()).device)
    reconstruction = model(sample)
    loss = criterion(reconstruction, sample)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    losses.append(loss.item())


def get_time():
    now = datetime.now()
    year = now.year
    month = now.month
    day = now.day
    hour = now.hour
    minute = now.minute
    time = f"{year}-{month}-{day}-{hour}-{minute}"
    return time


def save_model(model: nn.Module, epoch: int) -> None:
    models_directory = os.path.join("models", CURRENT_TIME)
    os.makedirs(models_directory, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(models_directory, f"{epoch}.pt"))


if __name__ == "__main__":
    CURRENT_TIME = get_time()
    train_model()
