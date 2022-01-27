import torch
from torch import nn
import torch.nn.functional as F

import math


class Memory(nn.Module):
    def __init__(self, dim, num_item=100) -> None:
        super(Memory, self).__init__()
        mempool = nn.Parameter(torch.Tensor(num_item, dim))
        self.mempool = self.init_memory(mempool)
        
    def init_memory(self, memory):
        stdv = 1. / math.sqrt(memory.size(1))
        memory.data.uniform_(-stdv, stdv)

        return memory
    
    def memorize(self, input):
        shape = input.shape
        
        # == GET QUERIES ==
        if len(shape) == 4:
            input = input.permute(0, 2, 3, 1)
        elif len(shape) == 5:
            input = input.permute(0, 2, 3, 4, 1)
            
        input = input.contiguous()
        query = input.view(-1, shape[1])
        
        # == GET ATTENTION VECTORS ==
        att = F.linear(query, self.mempool) # for LRP, use the attention vector before softmax
        att = F.softmax(att, dim=1)
        
        # == MEMORY SELECTION ==
        mempool_T = self.mempool.permute(1, 0)
        output = F.linear(att, mempool_T)
        
        # == RECOVER DIMENSIONALITY ==
        if len(shape) == 4:
            output = output.view(shape[0], shape[2], shape[3], shape[1])
            output = output.permute(0, 3, 1, 2)
        if len(shape) == 5:
            output = output.view(shape[0], shape[2], shape[3], shape[4], shape[1])
            output = output.permute(0, 4, 1, 2, 3)
            
        return output

    def forward(self, input):
        output = self.memorize(input)
        
        return output