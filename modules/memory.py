import torch
from torch import nn
import torch.nn.functional as F
import modules.layerx as nx

import math


class Memory(nn.Module):
    def __init__(self, dim, num_item, k) -> None:
        super(Memory, self).__init__()
        mempool = nn.Parameter(torch.Tensor(num_item, dim))
        self.init_memory(mempool)
        
        self.linear = nx.Linear(512, 20, bias=False)
        self.linear.weight = mempool
        
        self.k = k
        
    def init_memory(self, memory):
        stdv = 1. / math.sqrt(memory.size(1))
        memory.data.uniform_(-stdv, stdv)   
    
    def forward(self, input):
        shape = input.shape
        
        # == GET QUERIES ==
        input = input.permute(0, 2, 3, 1)
        
        input = input.contiguous()
        query = input.view(-1, shape[1])
        
        # == GET ATTENTION VECTORS ==
        att = self.linear(query)
        
        # == Top-K SELECTION ==
        val, idx = torch.topk(att, k=self.k, dim=1)
        if self.training:
            val = F.softmax(val, dim=1)
        else:
            val = torch.ones_like(val)
        attvec = torch.zeros_like(att).scatter_(1, idx, val)
        
        # == MEMORY SELECTION ==
        mempool_T = self.linear.weight.permute(1, 0)
        output = F.linear(attvec, mempool_T)
        
        # == RECOVER DIMENSIONALITY ==
        output = output.view(shape[0], shape[2], shape[3], shape[1])
        output = output.permute(0, 3, 1, 2)
        
        return output, attvec*att
    
    def relprop(self, R, alpha):
        R = self.linear.relprop(R, alpha)
        R = R.view(1, 32, 32, 512) # invert: input.view(-1, shape[1])
        R = R.permute(0, 3, 1, 2) # invert: input.permute(0, 2, 3, 1)
        
        return R
    
    def RAP_relprop(self, R):
        R = self.linear.RAP_relprop(R)
        R = R.view(1, 32, 32, 512) # invert: input.view(-1, shape[1])
        R = R.permute(0, 3, 1, 2) # invert: input.permute(0, 2, 3, 1)
        
        return R