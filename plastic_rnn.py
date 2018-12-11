import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from numpy import random
import torch.nn.functional as F

#ttype = torch.FloatTensor;         # For CPU
ttype = torch.cuda.FloatTensor;     # For GPU 


class RNN(nn.Module):
    def __init__(self, nin, nout, hebb=True, ln=True):
        super(RNN, self).__init__()
        self.h = Cell(nin, nout, hebb=hebb, ln=ln, nonlin=nn.ReLU)
        self.hebb = hebb
        self.nin  = nin
        self.nout = nout
    
    def forward(self, x, hid, hebb=(None)):
        ht, hp = self.h(x, hid, hebb[0])
        # import pdb; pdb.set_trace()
        return ht, (hp,)

    def initialZeroState(self):
        # Return an initialized, all-zero hidden state
        return Variable(torch.zeros(1, self.nout).type(ttype))

    def initialZeroHebb(self):
        # Return an initialized, all-zero Hebbian trace
        def zero_hebb():
            return Variable(torch.zeros(self.nout, self.nout).type(ttype)) + 1e-6
        return (
            zero_hebb() if self.hebb else None,
        )

class GRU(nn.Module):
    def __init__(self, nin, nout, hebb=set(), ln=True):
        super(GRU, self).__init__()
        self.h = Cell(nin, nout, hebb='h' in hebb, ln=ln, nonlin=nn.Tanh)
        self.z = Cell(nin, nout, hebb='z' in hebb, ln=ln, nonlin=nn.Sigmoid)
        self.r = Cell(nin, nout, hebb='r' in hebb, ln=ln, nonlin=nn.Sigmoid)
        self.hebb = hebb
        self.nin  = nin
        self.nout = nout
    
    def forward(self, x, hid, hebb=(None, None, None)):
        zt, zp = self.z(x, hid, hebb[0])
        rt, rp = self.r(x, hid, hebb[1])
        ht, hp = self.h(x, rt * hid, hebb[2])
        st = zt * hid + (-zt + 1) * ht
        return st, (zp, rp, hp)

    def initialZeroState(self):
        # Return an initialized, all-zero hidden state
        return Variable(torch.zeros(1, self.nout).type(ttype))

    def initialZeroHebb(self):
        # Return an initialized, all-zero Hebbian trace
        def zero_hebb():
            return Variable(torch.zeros(self.nout, self.nout).type(ttype)) + 1e-6
        return (
            zero_hebb() if 'z' in self.hebb else None,
            zero_hebb() if 'r' in self.hebb else None,
            zero_hebb() if 'h' in self.hebb else None,
        )



class Cell(nn.Module):
    def __init__(self, nin, nout, hebb=True, ln=True, nonlin=nn.Tanh):
        super(Cell, self).__init__()

        self.inp_lin = nn.Linear(nin, nout, bias=False)
        self.hid_lin = Plastic(nout, nout) if hebb else nn.Linear(nout, nout, bias=True)
        self.nonlin = nonlin()
        self.hebb = hebb
        if ln:
            self.ln = torch.nn.LayerNorm(nout).type(ttype)

    def forward(self, x, hid, hebb=None):
        assert (hebb is not None) == self.hebb

        wx = self.nonlin(self.inp_lin(x))
        wh = self.hid_lin(*(hid, hebb) if self.hebb else hid)

        wy = self.nonlin(self.ln(wx + wh))

        if self.hebb:
            hebb = self.hid_lin.update_hebb(hid, wh, hebb) # TODO OR (hid, wh)
            return wy, hebb

        return wy, None

class Plastic(nn.Module):
    def __init__(self, nin, nout):
        super(Plastic, self).__init__()

        self.w = nn.Parameter(.01 * torch.eye(nin, nout).type(ttype), requires_grad=True)   # The matrix of fixed (baseline) weights
        self.nonlin = nn.Tanh()
        self.alpha = nn.Parameter(.01 * torch.randn(nin, nout).type(ttype), requires_grad=True)
        self.b = nn.Parameter(.01 * torch.randn(nout).type(ttype), requires_grad=True)

        # predict lambda?
        self.lamda = nn.Parameter(.01 * torch.ones(1).type(ttype), requires_grad=True)  # The weight decay term / "learning rate" of plasticity - trainable, but shared across all connections

        self.pred_eta = nn.Parameter(.01 * torch.randn(nout, nout**1).type(ttype), requires_grad=True)
        self.pred_eta_b = nn.Parameter(.01 * torch.randn(nout).type(ttype), requires_grad=True)

        # self.pred_eta = nn.Parameter(.01 * torch.randn(nout, nout**1).type(ttype), requires_grad=True)

        # import pdb; pdb.set_trace()

    def forward(self, x, hebb):
        return x.mm(self.w  + (torch.mul(self.alpha, hebb)) ) + self.b
        
    def update_hebb(self, x, y, hebb):
        eta_hat = F.sigmoid(y.mm(self.pred_eta) + self.pred_eta_b)

        # hebb = torch.clamp( hebb + eta_hat.transpose(0,1) * torch.bmm(yin.unsqueeze(2), ymid.unsqueeze(1))[0], min=-1, max=1)
        
        # first simple
        # hebb = (1 - self.eta) * hebb + self.eta * torch.bmm(yin.unsqueeze(2), ymid.unsqueeze(1))[0] # bmm here is used to implement an outer product between yin and yout, with the help of unsqueeze (i.e. added empty dimensions)
        
        # oja's rule?
        # hebb = hebb + self.eta * torch.bmm(ymid.unsqueeze(2), (yin - ymid.mm(hebb)).unsqueeze(1)  )[0]
        
        # clipped hebbian
        # hebb = torch.clamp( hebb + self.eta * torch.bmm(yin.unsqueeze(2), ymid.unsqueeze(1))[0], min=-1, max=1)

        # backpropamine
        eta_hat = F.sigmoid(y.mm(self.pred_eta) + self.pred_eta_b)

        # hebb = torch.clamp( hebb + eta_hat.transpose(0,1) * torch.bmm(yin.unsqueeze(2), ymid.unsqueeze(1))[0], min=-1, max=1)
        hebb = (1-self.lamda) * hebb + eta_hat.transpose(0,1) * torch.bmm(x.unsqueeze(2), y.unsqueeze(1))[0]
        # hebb = torch.clamp( hebb + eta_hat.view(self.nhid, self.nhid) * torch.bmm(yin.unsqueeze(2), ymid.unsqueeze(1))[0], min=-1, max=1)
       
        self.eta_hat = eta_hat

        return hebb