import torch
from torch import nn
from torch.autograd import Variable

from .model import Model
from .load import testloader
from .utils import visualize

from glob import glob
from tqdm import tqdm
import os


class Interpret():
    def __init__(self, args) -> None:
        super(Interpret, self).__init__()
        self.device = torch.device(f'cuda:{args.cuda}' \
                                   if torch.cuda.is_available() \
                                   else 'cpu')
        
        self.log_path = f'{args.log_path}/{args.dataset}'
        os.makedirs(self.log_path, exist_ok=True)
        
        self.gt_label = f'{args.data_path}/{args.dataset}_gt.npy'
        self.args = args
        
    def run(self):
        pth = 'mem200_batch8_seeds0_clip20_run02-16_05:57PM.pth'
        
        print(f'interpret on {self.args.dataset}: {pth}...')
        
        net = Model(self.args.clip_length).to(self.device)
        net.load_state_dict(torch.load(self.log_path+'/'+pth,
                                       map_location=f'cuda:{self.args.cuda}'))
        
        MSE = nn.MSELoss().to(self.device)
        
        net.eval()
        
        frame_path = f'{self.args.data_path}/{self.args.dataset}/testing/frames'
        videos = sorted(glob(os.path.join(frame_path, '*')))
        
        # ! Do not use torch.no_grad()
        for i, vid in enumerate(tqdm(videos)):
            loader = testloader(data_path=vid,
                                num_workers=self.args.num_workers,
                                window=self.args.clip_length)

            for idx, frame in enumerate(loader):
                frame = Variable(frame).to(self.device)
                frame.requires_grad = True
                
                _, R = net(frame[:,:-3])
                
                if self.args.LRP:
                    relevance = net.relprop(R=R, alpha=1)
                elif self.args.RAP:
                    relevance = net.RAP_relprop(R=R)
                
                relevance = relevance[:,-3:].sum(dim=1, keepdim=True)
                heatmap = relevance.permute(0, 2, 3, 1).data.cpu().numpy()
                visualize(heatmap, self.log_path, i+1, idx+1)