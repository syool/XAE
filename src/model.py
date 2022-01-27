from torch import nn
from modules import Encoder, Decoder, Memory


class Model(nn.Module):
    def __init__(self, clip_length) -> None:
        super(Model, self).__init__()
        self.en = Encoder(clip_length)
        self.de = Decoder()
        self.me = Memory(512, num_item=20)

    def forward(self, x):
        z, skip = self.en(x)
        z = self.me(z)
        x = self.de(z, skip)

        return x