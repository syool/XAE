import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.backends import cudnn

from .model import Model
from .load import trainloader
from .utils import weights_init, seeds

import time
import os

class Train():
    def __init__(self, args) -> None:
        super(Train, self).__init__()
        self.device = torch.device(f'cuda:{args.cuda}' \
                                   if torch.cuda.is_available() \
                                   else 'cpu')
        cudnn.benchmark = False
        cudnn.deterministic = True
        seeds(args.seed)
        
        frame_path = f'{args.data_path}/{args.dataset}/training/frames'
        self.loader = trainloader(data_path=frame_path,
                                  batch=args.batch,
                                  num_workers=args.num_workers,
                                  window=args.clip_length)
        
        self.log_path = f'{args.log_path}/{args.dataset}'
        os.makedirs(self.log_path, exist_ok=True)
        
        self.args = args
        
    def run(self):
        print(f'train on {self.args.dataset}...')
        
        net = Model(self.args.clip_length).to(self.device)
        net.apply(weights_init)
        
        optimizer = optim.Adam(net.parameters(), lr=self.args.lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args.epochs)
        
        MSE = nn.MSELoss().to(self.device)
        
        net.train()
        for epoch in range(self.args.epochs):
            for i, frame in enumerate(self.loader):
                frame = Variable(frame).to(self.device)
                
                optimizer.zero_grad()
                
                output = net(frame[:,:-3]) # frame[:,:-3] -> frame clip
                loss = MSE(output, frame[:,-3:]) # frame[:,-3:] -> future frame
                
                loss.backward()
                optimizer.step()
                
                if i % 10 == 9:
                    print(f'{self.args.dataset}:',
                          f'Epoch {epoch+1}/{self.args.epochs}',
                          f'Batch {i+1}/{len(self.loader)}',
                          f'recon.: {loss.item():.6f}')
                
            scheduler.step()
            
            # torch.save(net.state_dict(), f'{self.log_path}/mem200_{epoch+1}.pth')
        
        ftime = time.strftime('%m-%d_%I:%M%p', time.localtime())
        picklef = f'mem200_batch{self.args.batch}_seeds{self.args.seed}_clip{self.args.clip_length}_run{ftime}'
        
        torch.save(net.state_dict(), f'{self.log_path}/{picklef}.pth')
        print(f'{self.args.dataset} training done')
    