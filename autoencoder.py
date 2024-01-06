import urllib.parse
from argparse import ArgumentParser

import torch
from pytorch_lightning import LightningModule, Trainer
from torch import nn
from torch.nn import functional as F
from modules import (
    EncoderBlock, ResNetEncoder, DecoderBlock, ResNetDecoder,
    resnet18_encoder, resnet18_decoder
)

class AE(LightningModule):

    def __init__(
        self,
        input_height: int,
        enc_type: str = "resnet18",
        first_conv: bool = False,
        maxpool1: bool = False,
        enc_out_dim: int = 512,
        latent_dim: int = 256,
        lr: float = 1e-4,
        **kwargs,
    ) -> None:
        """
        Args:
            input_height: height of the images
            enc_type: option between resnet18 or resnet50
            first_conv: use standard kernel_size 7, stride 2 at start or
                replace it with kernel_size 3, stride 1 conv
            maxpool1: use standard maxpool to reduce spatial dim of feat by a factor of 2
            enc_out_dim: set according to the out_channel count of
                encoder used (512 for resnet18, 2048 for resnet50)
            latent_dim: dim of latent space
            lr: learning rate for Adam
        """

        super().__init__()

        self.save_hyperparameters()

        self.lr = lr
        self.enc_out_dim = enc_out_dim
        self.latent_dim = latent_dim
        self.input_height = input_height

        valid_encoders = {
            "resnet18": {
                "enc": resnet18_encoder,
                "dec": resnet18_decoder,
            },
            "resnet50": {
                "enc": resnet50_encoder,
                "dec": resnet50_decoder,
            },
        }

        if enc_type not in valid_encoders:
            self.encoder = resnet18_encoder(first_conv, maxpool1)
            self.decoder = resnet18_decoder(self.latent_dim, self.input_height, first_conv, maxpool1)
        else:
            self.encoder = valid_encoders[enc_type]["enc"](first_conv, maxpool1)
            self.decoder = valid_encoders[enc_type]["dec"](self.latent_dim, self.input_height, first_conv, maxpool1)

        self.fc = nn.Linear(self.enc_out_dim, self.latent_dim)

    def from_pretrained(self, ckpt_path):
        return self.load_from_checkpoint(ckpt_path, strict=False)

    def forward(self, x):
        feats = self.encoder(x)
        z = self.fc(feats)
        return self.decoder(z)

    def step(self, batch, batch_idx):
        x, y = batch

        feats = self.encoder(x)
        z = self.fc(feats)
        x_hat = self.decoder(z)

        loss = F.mse_loss(x_hat, x, reduction="mean")

        return loss, {"loss": loss}

    def training_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        self.log_dict({f"train_{k}": v for k, v in logs.items()}, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        self.log_dict({f"val_{k}": v for k, v in logs.items()})
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

