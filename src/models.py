import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.activation = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.activation(x)
        return x

    def extract_notes(self, x):
        x = self.forward(x)
        x = torch.argmax(x, dim=1)
        return x


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.tconv1 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.activation = nn.ReLU()
        self.tconv2 = nn.ConvTranspose2d(32, 1, kernel_size=3, stride=1, padding=1)
        self.output = nn.Sigmoid()

    def forward(self, x):
        x = self.tconv1(x)
        x = self.activation(x)
        x = self.tconv2(x)
        return x


class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def extract_notes(self, x):
        return self.encoder.extract_notes(x)


class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def extract_notes(self, x):
        max = torch.argmax(x, dim=0)
        return max


MODELS = {
    "Identity": Identity,
    "Autoencoder": Autoencoder,
}