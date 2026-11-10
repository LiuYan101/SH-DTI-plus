"""Conditional noise loss, EMA and DDPM sampling for SH-DTI++."""
import numpy as np
import torch


def beta_schedule(config,device):
    c=config.diffusion
    return torch.from_numpy(np.linspace(c.beta_start,c.beta_end,c.num_diffusion_timesteps,dtype=np.float64)).float().to(device)


def noise_loss(model,sh,target,t,noise,betas):
    alpha=(1-betas).cumprod(0)[t].view(-1,1,1,1)
    xt=target*alpha.sqrt()+noise*(1-alpha).sqrt()
    predicted=model(torch.cat([sh,xt],dim=1),t.float())
    return (noise-predicted).square().sum(dim=(1,2,3)).mean()


class EMA:
    def __init__(self,model,rate=.999):
        self.rate=rate
        self.shadow={name:p.detach().clone() for name,p in model.named_parameters() if p.requires_grad}

    @torch.no_grad()
    def update(self,model):
        for name,p in model.named_parameters():
            if p.requires_grad:
                self.shadow[name].mul_(self.rate).add_(p,alpha=1-self.rate)

    @torch.no_grad()
    def apply(self,model):
        for name,p in model.named_parameters():
            if p.requires_grad:
                p.copy_(self.shadow[name])


def load_weights(model,path,device,use_ema=True):
    # Load only trusted local checkpoints; the original format is a state list.
    state=torch.load(path,map_location=device,weights_only=False)
    if isinstance(state,(list,tuple)):
        weights=state[0]
        ema=state[4] if len(state)>4 else None
    else:
        weights=state.get('model',state)
        ema=state.get('ema')
    weights={k.removeprefix('module.'):v for k,v in weights.items()}
    model.load_state_dict(weights,strict=True)
    if use_ema:
        if ema is None:
            raise ValueError('Checkpoint has no EMA parameters; use --no-ema if intentional.')
        ema={k.removeprefix('module.'):v for k,v in ema.items()}
        helper=EMA(model)
        if set(ema)!=set(helper.shadow):
            raise ValueError('EMA parameter keys do not match SH-DTI++.')
        helper.shadow=ema
        helper.apply(model)


@torch.no_grad()
def sample(model,sh,betas):
    x=torch.randn((sh.shape[0],6,*sh.shape[2:]),device=sh.device)
    alpha=torch.cat([betas.new_ones(1),(1-betas).cumprod(0)])
    for step in reversed(range(len(betas))):
        t=betas.new_full((len(x),),step)
        at,previous=alpha[step+1],alpha[step]
        beta=1-at/previous
        epsilon=model(torch.cat([sh,x],dim=1),t)
        clean=x/at.sqrt()-(1/at-1).sqrt()*epsilon
        # Tensor components are not clipped to an image-intensity range.
        mean=(previous.sqrt()*beta*clean+(1-beta).sqrt()*(1-previous)*x)/(1-at)
        noise=torch.randn_like(x)
        x=mean+(step>0)*beta.sqrt()*noise
    return x
