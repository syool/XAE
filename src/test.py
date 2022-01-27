import torch
from torch import nn
from torch.autograd import Variable

from .model import Model
from .load import testloader
from .utils import label_encapsule, psnr, score_norm

import numpy as np
import sklearn.metrics as skmetr
from glob import glob
from tqdm import tqdm
import os


class Test():
    def __init__(self, args) -> None:
        super(Test, self).__init__()
        self.device = torch.device(f'cuda:{args.cuda}' \
                                   if torch.cuda.is_available() \
                                   else 'cpu')
        
        self.log_path = f'{args.log_path}/{args.dataset}'
        os.makedirs(self.log_path, exist_ok=True)
        
        self.gt_label = f'{args.data_path}/{args.dataset}_gt.npy'
        self.args = args
        
    def run(self):
        pth = 'mem200_batch8_seeds0_clip20_run12-17_06:42PM_auc972.pth'
        
        print(f'test on {self.args.dataset}: {pth}...')
        
        net = Model(self.args.clip_length).to(self.device)
        net.load_state_dict(torch.load(self.log_path+'/'+pth,
                                       map_location=f'cuda:{self.args.cuda}'))
        
        MSE = nn.MSELoss().to(self.device)
        
        net.eval()
        
        frame_path = f'{self.args.data_path}/{self.args.dataset}/testing/frames'
        videos = sorted(glob(os.path.join(frame_path, '*')))
        
        labels = label_encapsule(np.load(self.gt_label).squeeze(),
                                 frame_path, self.args.clip_length)
        
        scores = []
        with torch.no_grad():
            for i, vid in enumerate(tqdm(videos)):
                loader = testloader(data_path=vid,
                                    num_workers=self.args.num_workers,
                                    window=self.args.clip_length)

                err_list = []
                for idx, frame in enumerate(loader):
                    frame = Variable(frame).to(self.device)
                    
                    output = net(frame[:,:-3])
                      
                    error = MSE(output[0], frame[0,-3:])
                    err_list = np.append(err_list, psnr(error.item()))
                    
                p = score_norm(err_list)
                scores = np.append(scores, p)
                # print(f'video {i+1}/{len(videos)} done')
            
        # final evaluation
        fpr, tpr, _ = skmetr.roc_curve(labels, scores, pos_label=0)
        auc = skmetr.auc(fpr, tpr)

        print(f'fianl auc: {auc}')