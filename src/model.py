from torch import nn
from modules import Encoderx, Decoder, Memoryx


class Model(nn.Module):
    def __init__(self, clip_length) -> None:
        super(Model, self).__init__()
        self.encoder = Encoderx(clip_length)
        self.decoder = Decoder()
        self.memory = Memoryx(512, 20, 5)

    def forward(self, x):
        z, skip = self.encoder(x)
        z = self.memory(z)
        x = self.decoder(z, skip)

        return x