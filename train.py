"""Train SH-DTI++ on paired SH/tensor patches."""
import argparse
import random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from sh_dti_plus.io import load_config
from sh_dti_plus.data import PatchDataset
from sh_dti_plus.model import SHDTIPlus
from sh_dti_plus.diffusion import EMA,beta_schedule,noise_loss


def main():
    p=argparse.ArgumentParser(description='Train SH-DTI++')
    p.add_argument('--config',default='configs/sh_dti_plus.yaml')
    p.add_argument('--data',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--resume',type=Path)
    p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--seed',type=int,default=1234)
    a=p.parse_args()
    c=load_config(a.config)
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(a.seed)
    if not c.model.use_se:
        raise ValueError('SH-DTI++ requires use_se: true.')
    a.output.mkdir(parents=True,exist_ok=True)
    if not a.resume and any(a.output.glob('*.pth')):
        raise FileExistsError('Output already has checkpoints; resume explicitly or use a new output directory.')
    loader=DataLoader(PatchDataset(a.data),batch_size=c.training.batch_size,shuffle=True,
                      num_workers=c.data.num_workers,pin_memory=a.device.startswith('cuda'))
    model=SHDTIPlus(c).to(a.device)
    optimizer=torch.optim.Adam(model.parameters(),lr=c.optim.lr,betas=(c.optim.beta1,.999),
        eps=c.optim.eps,weight_decay=c.optim.weight_decay,amsgrad=c.optim.amsgrad)
    ema=EMA(model,c.model.ema_rate)
    start_epoch=step=0
    if a.resume:
        state=torch.load(a.resume,map_location=a.device,weights_only=False)
        model.load_state_dict(state[0],strict=True)
        optimizer.load_state_dict(state[1])
        start_epoch,step=state[2:4]
        ema.shadow=state[4]
    betas=beta_schedule(c,a.device)
    def save(epoch):
        state=[model.state_dict(),optimizer.state_dict(),epoch,step,ema.shadow]
        torch.save(state,a.output/f'ckpt_{step}.pth')
        torch.save(state,a.output/'ckpt.pth')
    if step>=c.training.n_iters:
        print(f'Checkpoint already reached {step} steps.')
        return
    for epoch in range(start_epoch,c.training.n_epochs):
        for sh,target in loader:
            model.train()
            sh,target=sh.to(a.device),target.to(a.device)
            noise=torch.randn_like(target)
            t=torch.randint(0,len(betas),(len(sh)//2+1,),device=a.device)
            t=torch.cat([t,len(betas)-t-1])[:len(sh)]
            loss=noise_loss(model,sh,target,t,noise,betas)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if getattr(c.optim,'grad_clip',None) is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(),c.optim.grad_clip)
            optimizer.step();ema.update(model);step+=1
            if step==1 or step%100==0:
                print(f'SH-DTI++ step={step} loss={loss.item():.6f}',flush=True)
            if step==1 or step%c.training.snapshot_freq==0 or step>=c.training.n_iters:
                save(epoch)
            if step>=c.training.n_iters:return
    save(epoch)


if __name__=='__main__':
    main()
