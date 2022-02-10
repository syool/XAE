from torch import nn
from modules import Encoder, Decoder, Memory


class Model(nn.Module):
    def __init__(self, clip_length) -> None:
        super(Model, self).__init__()
        self.encoder = Encoder(clip_length)
        self.decoder = Decoder()
        self.memory = Memory(512, 20, 5)

    def forward(self, x):
        z, skip = self.encoder(x)
        z, att = self.memory(z)
        x = self.decoder(z, skip)

        return x, att
    
    def relprop(self, R, alpha):
        R = self.memory.relprop(R, alpha)
        R = self.encoder.relprop(R, alpha)
        
        return R
    
    def RAP_relprop(self, R):
        R = self.memory.RAP_relprop(R)
        R = self.encoder.RAP_relprop(R)
        
        return R