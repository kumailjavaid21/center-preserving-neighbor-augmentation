import torch
BUDGET=2400

def optimizer(g):
 opt=torch.optim.Adam(g.parameters(),lr=5e-5,betas=(.5,.999),weight_decay=0)
 return opt,torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=BUDGET,eta_min=1e-6)

def hinge(m,denominator):
 return torch.relu(torch.log((m+1e-8)/(denominator.detach()+1e-8)))

def component(m,ci,arm,view,p,b):
 sup=(1.5 if ci==0 else 1.)/4.5*torch.log(m+1e-8) if ci<4 else .05*torch.log1p(m/1e-6)
 h=hinge(m,p if view==0 else b) if ci<4 else m*0
 return .5*sup+(.25*((1.5 if ci==0 else 1.)/4.5)*h if arm=='ANCHORED' and ci<4 else 0),sup,h
