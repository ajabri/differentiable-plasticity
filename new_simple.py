# Differentiable plasticity: simple binary pattern memorization and reconstruction.
#
# Copyright (c) 2018 Uber Technologies, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

# This program is meant as a simple instructional example for differentiable plasticity. It is fully functional but not very flexible.

# Usage: python simple.py [rngseed], where rngseed is an optional parameter specifying the seed of the random number generator. 
# To use it on a GPU or CPU, toggle comments on the 'ttype' declaration below.



#######
'''

SGD is not bad
Sparsity?
Eta row-wise is not bad

Need to measure everything

Continuous stream with mixture?

Show loss decreases within episode? meta-learning


WRITE a PlasticBlock
    Linear + tanh

Changing mixture
Only first 10 labels

'''

import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from numpy import random
import torch.nn.functional as F
from torch import optim
import random
import sys
import pickle as pickle
import pdb
import time

import vae
import sklearn
import sklearn.decomposition

import sys



def info(type, value, tb):
    if hasattr(sys, 'ps1') or not sys.stderr.isatty():
    # we are in interactive mode or we don't have a tty-like
    # device, so we call the default hook
        sys.__excepthook__(type, value, tb)
    else:
        import traceback, pdb
        # we are NOT in interactive mode, print the exception...
        traceback.print_exception(type, value, tb)
        print
        # ...then start the debugger in post-mortem mode.
        # pdb.pm() # deprecated
        pdb.post_mortem(tb) # more "modern"

sys.excepthook = info


# Parsing command-line arguments
params = {}; params['rngseed'] = 0
parser = argparse.ArgumentParser()
parser.add_argument("--rngseed", type=int, help="random seed", default=0)
parser.add_argument("--lr", type=float, help="learning rate of Adam optimizer", default=3e-4)
parser.add_argument("--type", help="network type ('plastic' or 'nonplastic')", default='plastic')
parser.add_argument("--name", help="network type ('plastic' or 'nonplastic')", default='main')
parser.add_argument("--T", type=int, help="number of time steps between each pattern presentation (with zero input)", default=20)
parser.add_argument("--n-adapt", type=float, default=0.5, help="number of time steps between each pattern presentation (with zero input)")
parser.add_argument("--nhid", type=int, default=50, help="number of time steps between each pattern presentation (with zero input)")
parser.add_argument("--N", type=int, default=50000, help="number of time steps between each pattern presentation (with zero input)")
parser.add_argument("--rnn-type", type=str, default='GRU', help="GRU | RNN")

args = parser.parse_args(); argvars = vars(args); argdict =  { k : argvars[k] for k in argvars if argvars[k] != None }
params.update(argdict)


RNGSEED = 0             # Initial random seed - can be modified by passing a number as command-line argument


if len(sys.argv) == 2:
    RNGSEED = int(sys.argv[1])
    print("Setting RNGSEED to "+str(RNGSEED))
np.set_printoptions(precision=3)
np.random.seed(RNGSEED); random.seed(RNGSEED); torch.manual_seed(RNGSEED)


#ttype = torch.FloatTensor;         # For CPU
ttype = torch.cuda.FloatTensor;     # For GPU 

import visdom
exp_id = int(time.time()/100)
vis = visdom.Visdom(port=8095, env=str(params['name']) + ' (%s)' % exp_id)
import time
vis.text('', opts=dict(width=10000, height=1))


from plastic_rnn import GRU, RNN

class PlasticPredictor(nn.Module):
    def __init__(self, nin, nhid, nout, hebb='z,r,h', ln=True):
        super(PlasticPredictor, self).__init__()

        # self.inp = nn.Identity() # or Plastic
        if params['rnn_type'] == 'GRU':
            self.rnn = GRU(nin, nhid, hebb=set(hebb.split(',')), ln=True)
        else:
            self.rnn = RNN(nin, nhid, hebb=params['type']=='plastic', ln=True)
        self.out = nn.Linear(nhid, nout)

    def forward(self, x, hid, hebb=None):
        # x = self.inp(x)
        hid, hebb = self.rnn(x, hid, hebb)
        y = self.out(hid)

        # HACK
        self.eta_hat = torch.zeros(1) if not hasattr(self.rnn.h.hid_lin, 'eta_hat') else self.rnn.h.hid_lin.eta_hat
        self.lamda = torch.zeros(1) if not hasattr(self.rnn.h.hid_lin, 'lamda') else self.rnn.h.hid_lin.lamda

        return hid, hebb, y

    def initialZeroState(self):
        # Return an initialized, all-zero hidden state
        return self.rnn.initialZeroState()

    def initialZeroHebb(self):
        # Return an initialized, all-zero Hebbian trace
        return self.rnn.initialZeroHebb()

class NETWORK(nn.Module):
    def __init__(self):
        super(NETWORK, self).__init__()
        # Notice that the vectors are row vectors, and the matrices are transposed wrt the usual order, following apparent pytorch conventions
        # Each *column* of w targets a single output neuron

        self.nhid = params['nhid']

        self.w0 = Variable(0.01 * torch.randn(3, self.nhid).type(ttype), requires_grad=True)
        self.b0 = Variable(0.01 * torch.randn(self.nhid).type(ttype), requires_grad=True)

        self.w = Variable(.01 * torch.eye(self.nhid, self.nhid).type(ttype), requires_grad=True)   # The matrix of fixed (baseline) weights

        self.w1 = Variable(0.01 * torch.randn(self.nhid, 1).type(ttype), requires_grad=True)
        self.b1 = Variable(0.01 * torch.randn(1).type(ttype), requires_grad=True)

        self.alpha = Variable(.01 * torch.randn(self.nhid, self.nhid).type(ttype), requires_grad=True)  # The matrix of plasticity coefficients
        self.eta = Variable(.01 * torch.ones(1).type(ttype), requires_grad=True)  # The weight decay term / "learning rate" of plasticity - trainable, but shared across all connections

        self.pred_eta = Variable(.01 * torch.randn(self.nhid, self.nhid**1).type(ttype), requires_grad=True)
        self.pred_eta_b = Variable(.01 * torch.randn(self.nhid).type(ttype), requires_grad=True)

        self.ln = torch.nn.LayerNorm(self.nhid).type(ttype)

    
    def forward(self, input, yin, hebb, E=None):
        # Run the network for one timestep

        nonlin = F.relu

        # 0
        # input = nonlin(input.mm(self.w0))
        # yout = nonlin( yin.mm(self.w + torch.mul(self.alpha, hebb)) + input )
        # hebb = (1 - self.eta) * hebb + self.eta * torch.bmm(yin.unsqueeze(2), yout.unsqueeze(1))[0] # bmm here is used to implement an outer product between yin and yout, with the help of unsqueeze (i.e. added empty dimensions)

        # 0.5: apply hebbian learning where input also get hebbian weights
        # input = nonlin(input.mm(self.w0 + torch.mul(self.alpha, hebb)))
        # yout = nonlin( yin.mm(self.w + torch.mul(self.alpha, hebb)) + input )

        # 1: apply hebbian learning where yout is before adding input
        i_t = nonlin( input.mm(self.w0) )

        # eta_hat = F.tanh(i_t.mm(self.pred_eta) + self.pred_eta_b) * 0 + 1
        # ymid = yin.mm( eta_hat.transpose(0,1) * (self.w + torch.mul(self.alpha, hebb)) )

        ymid = yin.mm(self.w  + (torch.mul(self.alpha, hebb)) )
        # ymid = yin.mm( (torch.mul(self.alpha, hebb)) )

        # ymid = yin.mm( self.w + torch.mul(self.alpha, hebb) )
        
        yout = nonlin(self.ln(ymid + i_t + self.b0))

        # ymid = yout
        
        # first simple
        # hebb = (1 - self.eta) * hebb + self.eta * torch.bmm(yin.unsqueeze(2), ymid.unsqueeze(1))[0] # bmm here is used to implement an outer product between yin and yout, with the help of unsqueeze (i.e. added empty dimensions)
        
        # oja's rule?
        # hebb = hebb + self.eta * torch.bmm(ymid.unsqueeze(2), (yin - ymid.mm(hebb)).unsqueeze(1)  )[0]
        
        # clipped hebbian
        # hebb = torch.clamp( hebb + self.eta * torch.bmm(yin.unsqueeze(2), ymid.unsqueeze(1))[0], min=-1, max=1)

        # backpropamine
        eta_hat = F.sigmoid(ymid.mm(self.pred_eta) + self.pred_eta_b)
        # hebb = torch.clamp( hebb + eta_hat.transpose(0,1) * torch.bmm(yin.unsqueeze(2), ymid.unsqueeze(1))[0], min=-1, max=1)
        hebb = (1-self.eta) * hebb + eta_hat.transpose(0,1) * torch.bmm(yin.unsqueeze(2), ymid.unsqueeze(1))[0]
        # hebb = torch.clamp( hebb + eta_hat.view(self.nhid, self.nhid) * torch.bmm(yin.unsqueeze(2), ymid.unsqueeze(1))[0], min=-1, max=1)
       
        self.eta_hat = eta_hat
        # print(self.eta, params['type'])
        # print(hebb.mean().item())

        # import pdb; pdb.set_trace()        
        if params['type'] == 'nonplastic':
            hebb = 0
        # hebb = 0

        # hebbian trace
        # E = (1 - self.eta) * E + self.eta * torch.bmm(yin.unsqueeze(2), ymid.unsqueeze(1))[0] 
        # hebb = (1 - self.eta) * hebb + self.eta * torch.bmm(yin.unsqueeze(2), ymid.unsqueeze(1))[0] 

        # # bmm here is used to implement an outer product between yin and yout, with the help of unsqueeze (i.e. added empty dimensions)

        # 2: top-down modulation from error?
        
        out = yout.mm(self.w1) + self.b1

        return yout, hebb, out

    def initialZeroState(self):
        # Return an initialized, all-zero hidden state
        return Variable(torch.zeros(1, self.nhid).type(ttype))

    def initialZeroHebb(self):
        # Return an initialized, all-zero Hebbian trace
        return Variable(torch.zeros(self.nhid, self.nhid).type(ttype)) + 1e-6


# net = NETWORK()
net = PlasticPredictor(nin=3, nhid=params['nhid'], nout=1, hebb=params['type'], ln=True).cuda()

optimizer = torch.optim.Adam(net.parameters(), lr=params['lr'])
import pdb; pdb.set_trace
# optimizer = torch.optim.Adam([net.w, net.alpha, net.eta, net.pred_eta,
#     net.b1, net.b0, net.pred_eta_b,
#     net.w0, net.w1], lr=params['lr'])

# optimizer = torch.optim.SGD([net.w, net.alpha, net.eta,
#     net.b1, net.b0, net.pred_eta, net.pred_eta_b,
#     net.w0, net.w1],
#     lr=0.01)

total_loss = 0.0; all_losses = []
print_every = 100
nowtime = time.time()


# functions

# sine, box, line
# mixture

ll = []

for numiter in range(params['N']):
    # Initialize network for each episode


    # amplitude
    amp = np.random.random() * 10 - 5
    # amp = 1

    # phase
    ph = 1 * np.random.random() * np.pi/2 + 1
    xx = 1 * np.random.random() * 4 - 2
    yy = 0 # 1 * np.random.random() * 4 - 2
    # ph = 0

    _T = T = 30
    
    # extrapolation validation
    if np.random.random() < 0.0:
        T *= 2
    
    n_adapt = int(_T * params['n_adapt'])

    X = np.random.random() * 10 + 10
    if _T != T:
        X *= 2

    inputs = torch.arange(0, T).unsqueeze(-1).unsqueeze(-1).cuda() / T * X - (X/2)
    # inputs = inputs[torch.randperm(inputs.numel())]    
    # inputs = (torch.sort(torch.rand(T))[0]).unsqueeze(-1).unsqueeze(-1).cuda() 
    # inputs = ((inputs - inputs.min()) / (inputs.max() - inputs.min())) * X - X//2
    
    target = amp * torch.sin(ph*inputs + xx) + yy

    # Generate the inputs and target pattern for this episode
    # inputs, target = generateInputsAndTarget()

    # Run the episode!
    for it in range(1):
        out = []
        etas = []
        ys = []
        preds = []
        optimizer.zero_grad()
    
        hebb = net.initialZeroHebb()
        y = net.initialZeroState()
        err = torch.cat([inputs[0], inputs[0]]) * 0
        # err[-1] = 0

        for numstep in range(T):
            y, hebb, pred = net(Variable(torch.cat([inputs[numstep], err],dim=0).transpose(0,1), requires_grad=False), y, hebb)
            # err =  #* 0
            err = torch.cat([target[numstep], pred - target[numstep]]) * 1
            # import pdb; pdb.set_trace()

            if numstep > n_adapt: # or np.random.random() > 0.5:
                # err[0] = 0
                # err[1] = 0
                # print(pred)
                err[0] = pred
                # err[1] = 

            err[1] = 0

            # import pdb; pdb.set_trace()
            out.append(pred)
            etas.append(net.eta_hat)
            ys.append(y)

        # Compute loss for this episode (last step only)
        # import pdb; pdb.set_trace()
        # loss = (torch.stack(out) - Variable(target, requires_grad=False))[T//2:].pow(2).mean()
        loss = (torch.stack(out) - Variable(target, requires_grad=False))[n_adapt:].pow(2).mean()
        ll.append(loss.item())

        # loss += 0.0001 * sum([e.abs().mean() for e in etas]) / len(etas)
        eee = sum([e.abs()/ e.sum() for e in etas]) / len(etas)
        # import pdb; pdb.set_trace()
        # loss += 0.0001 * ((eee + 1e-6).log() * (eee + 1e-6)).sum()

        # if np.random.random() > 0.001:
            # print()
            # print(sum([e.abs().mean() for e in etas]).item() / len(etas))
            # print(sum([(e>0.1).float().mean() for e in etas]).item() / len(etas))
            # print()
        # loss += net.eta.norm()
        # Apply backpropagation to adapt basic weights and plasticity coefficients

        if T == _T:
            loss.backward()
            # print
            (torch.nn.utils.clip_grad_norm_(optimizer.param_groups[0]['params'], 10, norm_type=2))
            optimizer.step()


    # That's it for the actual algorithm!
    # Print statistics, save files
    lossnum = loss.data[0].item()   # Saved loss is the actual learning loss (MSE)
    #to = target.cpu().numpy(); yo = torch.stack(out).data.cpu().numpy(); z = (yo - to) ** 2; lossnum = np.mean(z[:])  # Saved loss is the error rate
    

    total_loss += lossnum
    if (numiter+1) % print_every == 0:
        # import pdb; pdb.set_trace()
        sz = min(50, len(ll))
        box = np.ones(sz)/sz
        lll = torch.Tensor(ll[::10])
        lc = np.convolve(lll, box, mode='valid')

        vis.line(Y=lc, X=torch.arange(lc.shape[0]),
            win=str(exp_id) + 'learning_curve',
            opts=dict(title='Training Curve', width=400, height=300))

        vis.line(
            X=torch.cat([inputs.squeeze(-1), inputs.squeeze(-1)], dim=-1),
            Y=torch.cat([target.squeeze(-1), torch.cat(out)], dim=-1),
            win=str(exp_id) + str((numiter+1) % 3) + str(T==_T),
            opts=dict(title=str(exp_id) + str((numiter+1) % 3) + str(T==_T)))

        # XX = torch.cat([torch.cat([inputs.squeeze(-1), inputs.squeeze(-1)], dim=0), torch.cat([target.squeeze(-1), torch.cat(out)], dim=0)], dim=1)
        # YY = torch.cat([torch.ones(target.squeeze(-1).shape), torch.ones(target.squeeze(-1).shape) + 1], dim=0)
        # vis.scatter(Y=YY, X=XX, win='scatter'+str(exp_id) + str((numiter+1) % 3), opts=dict(title=str(exp_id) + str((numiter+1) % 3)))

        pca = sklearn.decomposition.PCA(n_components=2)
        YY = torch.cat(ys).detach().cpu().numpy()
        pca.fit(YY)
        XX = pca.transform(YY)

        vis.scatter(X=XX,
            win='pca' + str(exp_id) + str((numiter+1) % 3) + str(T==_T),
            opts=dict(
                markercolor=(np.arange(T)/T * 255).astype(np.int),
                title='pca' + str(exp_id) + str((numiter+1) % 3) + str(T==_T)
            ))

        vis.scatter(X=np.concatenate([XX,  (np.arange(T)[:,None] - T/2) /T], axis=-1),
            win='pca3d' + str(exp_id) + str((numiter+1) % 3) + str(T==_T),
            opts=dict(
                markercolor=(np.arange(T)/T * 255).astype(np.int),
                title='pca' + str(exp_id) + str((numiter+1) % 3) + str(T==_T)
            ))

        # import pdb; pdb.set_trace()

        print((numiter, "===="))
        print(target.cpu().numpy()[-10:].squeeze())   # Target pattern to be reconstructed
        # print(inputs.cpu().numpy()[numstep][0][-10:])  # Last input contains the degraded pattern fed to the network at test time
        print(torch.stack(out).data.cpu().numpy()[-10:].squeeze())   # Final output of the network
        previoustime = nowtime
        nowtime = time.time()
        print("Time spent on last", print_every, "iters: ", nowtime - previoustime)
        total_loss /= print_every
        all_losses.append(total_loss)
        print("Mean loss over last", print_every, "iters:", total_loss)
        # print("Eta: ", net.lamda.item(), "mean Alpha", net.alpha.mean().item())
        print("Mean Eta: ", sum([e.abs().mean() for e in etas]).item() / len(etas), "Eta sparsity: ", sum([(e>0.1).float().mean() for e in etas]).item() / len(etas))
        # print(sum([e.abs()/ e.sum() for e in etas]) / len(etas))

        vis.text('<br>'.join(["%s" % i for i in all_losses]), win='log' + str(exp_id),
            opts=dict(width=300, height=300))

        total_loss = 0



