from torch import nn
import modules.layerx as nx


class Encoder(nn.Module):
    def __init__(self, clip_length) -> None:
        super(Encoder, self).__init__()
        def block(c_in, c_out):
            return nx.Sequential(
                nx.Conv2d(c_in, c_out, 3, stride=1, padding=1, bias=False),
                nx.BatchNorm2d(c_out),
                nx.ReLU(),
                
                nx.Conv2d(c_out, c_out, 3, stride=1, padding=1, bias=False),
                nx.BatchNorm2d(c_out),
                nx.ReLU()
            )
        def block_(c_in, c_out):
            return nx.Sequential(
                nx.Conv2d(c_in, c_out, 3, stride=1, padding=1, bias=False),
                nx.BatchNorm2d(c_out),
                nx.ReLU(),
                nx.Conv2d(c_out, c_out, 3, stride=1, padding=1, bias=False)
            )
            
        self.conv1 = block(clip_length*3, 64)
        self.pool1 = nx.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = block(64, 128)
        self.pool2 = nx.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3 = block(128, 256)
        self.pool3 = nx.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv4 = block_(256, 512)

    def forward(self, x):
        c1 = self.conv1(x)
        p1 = self.pool1(c1)
        
        c2 = self.conv2(p1)
        p2 = self.pool2(c2)
        
        c3 = self.conv3(p2)
        p3 = self.pool3(c3)
        
        z = self.conv4(p3)

        return z, (c1, c2, c3)
    
    def relprop(self, R, alpha):
        R = self.conv4.relprop(R, alpha)
        
        R = self.pool3.relprop(R, alpha)
        R = self.conv3.relprop(R, alpha)
        
        R = self.pool2.relprop(R, alpha)
        R = self.conv2.relprop(R, alpha)
        
        R = self.pool1.relprop(R, alpha)
        R = self.conv1.relprop(R, alpha)
        
        return R