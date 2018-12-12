#######
'''

SGD is not bad
Sparsity?
Eta row-wise is not bad

[DOING] Need to measure everything

Continuous stream with mixture?

[DONE] Show loss decreases within episode? meta-learning


[DONE] WRITE a PlasticBlock
    Linear + tanh

Changing mixture
[Done] Only first 10 labels

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
import os

import sys
import sklearn
import sklearn.decomposition



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
parser.add_argument("--reload_vae", help="", default='')
parser.add_argument("--reload", help="", default='')
parser.add_argument("--dataset", help="", default='balls')
parser.add_argument("--T", type=int, help="number of time steps between each pattern presentation (with zero input)", default=20)
parser.add_argument("--n-adapt", type=float, default=0.5, help="number of time steps between each pattern presentation (with zero input)")
parser.add_argument("--nhid", type=int, default=50, help="number of time steps between each pattern presentation (with zero input)")

args = parser.parse_args(); argvars = vars(args); argdict =  { k : argvars[k] for k in argvars if argvars[k] != None }
params.update(argdict)

import visdom
vis = visdom.Visdom(port=8095, env=str(params['name']) + str(time.time()))
import time
exp_id = int(time.time())
vis.text('', opts=dict(width=10000, height=1))


RNGSEED = 0             # Initial random seed - can be modified by passing a number as command-line argument

if len(sys.argv) == 2:
    RNGSEED = int(sys.argv[1])
    print("Setting RNGSEED to "+str(RNGSEED))
np.set_printoptions(precision=3)
np.random.seed(RNGSEED); random.seed(RNGSEED); torch.manual_seed(RNGSEED)


#ttype = torch.FloatTensor;         # For CPU
ttype = torch.cuda.FloatTensor;     # For GPU 

bce_loss = nn.BCELoss().type(ttype)


class GRU(nn.Module):
    def __init__(self, nin, nout, hebb=set(), ln=True):
        super(GRU, self).__init__()
        self.h = Cell(nin, nout, hebb='h' in hebb, ln=ln, nonlin=nn.Tanh)
        self.z = Cell(nin, nout, hebb='z' in hebb, ln=ln, nonlin=nn.Sigmoid)
        self.r = Cell(nin, nout, hebb='r' in hebb, ln=ln, nonlin=nn.Sigmoid)
        self.nin  = nin
        self.nout = nout
    
    def forward(self, x, hid, hebb=(None, None, None)):
        zt, zp = self.z(x, hid, hebb)
        rt, rp = self.r(x, hid, hebb)
        ht, hp = self.h(x, rt * hid, hebb)
        st = zt * hid + (-zt + 1) * ht
        return st, (zp, rp, hp)

    def initialZeroState(self):
        # Return an initialized, all-zero hidden state
        return Variable(torch.zeros(1, self.nhid).type(ttype))

    def initialZeroHebb(self):
        # Return an initialized, all-zero Hebbian trace
        return Variable(torch.zeros(self.nout, self.nout).type(ttype)) + 1e-6


class Cell(nn.Module):
    def __init__(self, nin, nout, hebb=True, ln=True, nonlin=nn.Tanh):
        super(Cell, self).__init__()

        self.inp_lin = nn.Linear(nin, nout)
        self.hid_lin = Plastic(nout, nout) if hebb else nn.Linear(nout, nout)
        self.nonlin = nonlin()
        self.hebb = hebb
        if ln:
            self.ln = torch.nn.LayerNorm(nin).type(ttype)

    def forward(self, x, hid, hebb=None):
        assert (hebb is not None) == self.hebb

        wx = self.inp_lin(x)
        wh = self.hid_lin((hid, hebb) if self.hebb else hid)

        wy = self.nonlin(self.ln(wx + wh))

        if self.hebb:
            hebb = self.hid_lin.update_hebb(hid, wy) # TODO OR (hid, wh)
            return wy, hebb

        return wy, None

class Plastic(nn.Module):
    def __init__(self, nin, nout):
        super(Plastic, self).__init__()

        self.w = nn.Parameter(.01 * torch.eye(nin, nout).type(ttype), requires_grad=True)   # The matrix of fixed (baseline) weights
        self.nonlin = nn.Tanh()
        self.alpha = nn.Parameter(.01 * torch.randn(nin, nout).type(ttype), requires_grad=True)

        # predict lambda?
        self.lamda = nn.Parameter(.01 * torch.ones(1).type(ttype), requires_grad=True)  # The weight decay term / "learning rate" of plasticity - trainable, but shared across all connections

        self.pred_eta = nn.Parameter(.01 * torch.randn(nout, nout**1).type(ttype), requires_grad=True)
        self.pred_eta_b = nn.Parameter(.01 * torch.randn(nout).type(ttype), requires_grad=True)

        # self.pred_eta = nn.Parameter(.01 * torch.randn(nout, nout**1).type(ttype), requires_grad=True)

        # import pdb; pdb.set_trace()

    def forward(self, x, hebb):
        return x.mm(self.w  + (torch.mul(self.alpha, hebb)) )
        
    def update_hebb(self, x, y):
        eta_hat = F.sigmoid(y.mm(self.pred_eta) + self.pred_eta_b)

        # hebb = torch.clamp( hebb + eta_hat.transpose(0,1) * torch.bmm(yin.unsqueeze(2), ymid.unsqueeze(1))[0], min=-1, max=1)
        
        # first simple
        # hebb = (1 - self.eta) * hebb + self.eta * torch.bmm(yin.unsqueeze(2), ymid.unsqueeze(1))[0] # bmm here is used to implement an outer product between yin and yout, with the help of unsqueeze (i.e. added empty dimensions)
        
        # oja's rule?
        # hebb = hebb + self.eta * torch.bmm(ymid.unsqueeze(2), (yin - ymid.mm(hebb)).unsqueeze(1)  )[0]
        
        # clipped hebbian
        # hebb = torch.clamp( hebb + self.eta * torch.bmm(yin.unsqueeze(2), ymid.unsqueeze(1))[0], min=-1, max=1)

        # backpropamine
        eta_hat = F.sigmoid(ymid.mm(self.pred_eta) + self.pred_eta_b)

        # hebb = torch.clamp( hebb + eta_hat.transpose(0,1) * torch.bmm(yin.unsqueeze(2), ymid.unsqueeze(1))[0], min=-1, max=1)
        hebb = self.lamda * hebb + eta_hat.transpose(0,1) * torch.bmm(x.unsqueeze(2), y.unsqueeze(1))[0]
        # hebb = torch.clamp( hebb + eta_hat.view(self.nhid, self.nhid) * torch.bmm(yin.unsqueeze(2), ymid.unsqueeze(1))[0], min=-1, max=1)
       
        self.eta_hat = eta_hat

        return hebb

class PlasticPredictor(nn.Module):
    def __init__(self, nin, nhid, nout, hebb='h,z,r', ln=True):
        self.inp = nn.Identity() # or Plastic
        self.rnn = GRU(nin, nhid, hebb=set(hebb.split(',')), ln=True)
        self.out = nn.Linear(nhid, nout)

    def forward(self, x, hid, hebb=None):
        x = self.inp(x)
        hid, hebb = self.rnn(x, hid, hebb)
        y = self.out(hid)

        return hid, hebb, out

class PlasticPixelPredictor(nn.Module):
    def __init__(self, nin, nhid, nout, hebb='h,z,r', ln=True):
        self.rnn = GRU(nin, nhid, hebb=set(hebb.split(',')), ln=True)
        self.out = nn.Linear(nhid, nout)
        
        self.enc = vae.encoder()
        self.dec = vae.decoder()

    def forward(self, x, hid, hebb=None):
        x = self.enc(x)
        hid, hebb = self.rnn(x, hid, hebb)
        y = self.out(hid)

        out = self.dec(y)

        return hid, hebb, out

class NETWORK(nn.Module):
    def __init__(self):
        super(NETWORK, self).__init__()
        # Notice that the vectors are row vectors, and the matrices are transposed wrt the usual order, following apparent pytorch conventions
        # Each *column* of w targets a single output neuron

        self.nhid = 300

        self.w0 = Variable(0.01 * torch.randn(3, self.nhid).type(ttype), requires_grad=True)
        self.b0 = Variable(0.01 * torch.randn(self.nhid).type(ttype), requires_grad=True)

        self.w = Variable(.01 * torch.eye(self.nhid, self.nhid).type(ttype), requires_grad=True)   # The matrix of fixed (baseline) weights

        self.w1 = Variable(0.01 * torch.randn(self.nhid, self.nhid).type(ttype), requires_grad=True)
        self.b1 = Variable(0.01 * torch.randn(self.nhid).type(ttype), requires_grad=True)

        self.alpha = Variable(.01 * torch.randn(self.nhid, self.nhid).type(ttype), requires_grad=True)  # The matrix of plasticity coefficients
        self.eta = Variable(.01 * torch.ones(1).type(ttype), requires_grad=True)  # The weight decay term / "learning rate" of plasticity - trainable, but shared across all connections

        self.pred_eta = Variable(.01 * torch.randn(self.nhid, self.nhid**1).type(ttype), requires_grad=True)
        self.pred_eta_b = Variable(.01 * torch.randn(self.nhid).type(ttype), requires_grad=True)

        self.ln = torch.nn.LayerNorm(self.nhid).type(ttype)

        self.enc = vae.encoder().type(ttype)
        self.dec = vae.decoder().type(ttype)

        self.eta_hat = 0
    
    def get_params(self):
        return [self.enc, self.dec, self.w, self.b0, self.w1, self.b1, self.alpha, self.pred_eta, self.pred_eta_b, self.eta, self.eta_hat, self.ln]

    def load_params(self, p):
        self.enc, self.dec, self.w, self.b0, self.w1, self.b1, self.alpha, self.pred_eta, self.pred_eta_b, self.eta, self.eta_hat, self.ln = p

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
        i_t = self.enc(input)
        # import pdb; pdb.set_trace()
        # i_t = nonlin( input.mm(self.w0) )

        # eta_hat = F.tanh(i_t.mm(self.pred_eta) + self.pred_eta_b) * 0 + 1
        # ymid = yin.mm( eta_hat.transpose(0,1) * (self.w + torch.mul(self.alpha, hebb)) )

        ymid = yin.mm(self.w  + (torch.mul(self.alpha, hebb)) )
        # ymid = yin.mm( (torch.mul(self.alpha, hebb)) )

        # ymid = yin.mm( self.w + torch.mul(self.alpha, hebb) )
        
        yout = nonlin( self.ln(ymid + i_t + self.b0))
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

        out = self.dec(out)
        # import pdb; pdb.set_trace()

        return yout, hebb, out

    def initialZeroState(self):
        # Return an initialized, all-zero hidden state
        return Variable(torch.zeros(1, self.nhid).type(ttype))

    def initialZeroHebb(self):
        # Return an initialized, all-zero Hebbian trace
        return Variable(torch.zeros(self.nhid, self.nhid).type(ttype)) + 1e-6


net = NETWORK()

# import pdb; pdb.set_trace()

##############

if os.path.exists(params['reload_vae']) or os.path.exists(params['reload']):
    if os.path.exists(params['reload']):
        reloaded = torch.load(params['reload'])
        net.load_params(reloaded)

    if os.path.exists(params['reload_vae']):
        reloaded = list(torch.load(params['reload_vae']))

        net.enc, net.dec = reloaded

    optimizer = torch.optim.Adam([net.alpha, net.eta,
        net.pred_eta, net.pred_eta_b, #,
        net.b1, net.b0, 
        net.w0, net.w1, net.w], lr=params['lr'])

    # import pdb; pdb.set_trace()
else:
    optimizer = torch.optim.Adam([net.w, net.alpha, net.eta, net.pred_eta,
        net.b1, net.b0, net.pred_eta_b,# net.ln
        *list(net.enc.parameters()), *list(net.dec.parameters()),
        net.w0, net.w1], lr=params['lr'])

# optimizer = torch.optim.SGD([net.w, net.alpha, net.eta,
#     net.b1, net.b0, net.pred_eta, net.pred_eta_b,
#     net.w0, net.w1],
#     lr=0.01)

total_loss = 0.0; all_losses = []
print_every = 100
nowtime = time.time()


import scipy
import scipy.io

if params['dataset'] == 'mnist':
    mm1 = np.load('/data/ajabri/moving_mnist/mnist_test_seq_28.npy')

    hw = 16
    mm = np.ndarray((*mm1.shape[:-2], hw, hw))
    for t in range(mm1.shape[0]):
        for n in range(mm1.shape[1]):
            mm[t][n] = scipy.misc.imresize(mm1[t][n], (hw, hw))

elif params['dataset'] == 'balls':
    mm1 = scipy.io.loadmat('/data/ajabri/moving_mnist/bouncing_balls_training_data.mat')
    mm1 = mm1['Data'][0]
    hw = int(np.sqrt(mm1[0].shape[1]))
    mm = np.ndarray((mm1.shape[0], mm1[0].shape[0], hw, hw))

    for n in range(mm1.shape[0]):
        mm[n] = mm1[n].astype(mm.dtype).reshape(mm1[n].shape[0], hw, hw)
    mm = mm.transpose(1, 0 , 2, 3)
    
    mm = mm[:params['T']]
else:
    assert False, 'invalid dataset'

mm -= mm.min()
mm /= mm.max()
mm = torch.from_numpy(mm)


saved = False

n_pretrain = 10

import os
if os.path.exists(params['reload_vae']) or os.path.exists(params['reload']):
    n_pretrain = -1

ll = []

for e in range(100):
    torch.save(net.get_params(), '/data/ajabri/moving_mnist/checkpoints/%s_%s.pth' % (params['name'], exp_id))

    for numiter in range(mm.shape[1]-1):
    # for numiter in range(100):
        # Initialize network for each episode

        T = mm.shape[0]
        n_adapt = int(T * params['n_adapt'])


        # Generate the inputs and target pattern for this episode
        # inputs, target = generateInputsAndTarget()

        # Run the episode!
        out = []
        etas = []
        preds = []
        ys = []
        optimizer.zero_grad()

        
        if e > n_pretrain:
            if not saved and not os.path.exists(params['reload_vae']):
                torch.save( {net.dec, net.enc}, '/data/ajabri/moving_mnist/vae.pth')
                saved = True

            inputs = mm[:, numiter].type(ttype)
            target = mm[:, numiter+1].type(ttype)

            hebb = net.initialZeroHebb()
            y = net.initialZeroState()
            for numstep in range(T):
                if numstep > T // 2:
                    y, hebb, pred = net(Variable(pred, requires_grad=False), y, hebb)
                else:
                    y, hebb, pred = net(Variable(inputs[numstep].unsqueeze(0).unsqueeze(0), requires_grad=False), y, hebb)


                out.append(pred)
                etas.append(net.eta_hat)
                ys.append(y)

        else:
            idxs = torch.randperm(mm.shape[1]-1)[:100]
            inputs = mm[:, idxs].type(ttype)
            target = mm[:, idxs+1].type(ttype)

            for numstep in range(T):
                out.append(net.dec(net.enc(Variable(inputs[numstep].unsqueeze(1), requires_grad=False))))

            # import pdb; pdb.set_trace()
        loss = bce_loss(torch.stack(out).squeeze()[:], Variable(target, requires_grad=False)[:])
        ll.append(loss)

        # Apply backpropagation to adapt basic weights and plasticity coefficients
        loss.backward()
        # print
        (torch.nn.utils.clip_grad_norm_(optimizer.param_groups[0]['params'], 10, norm_type=2))
        optimizer.step()


        # That's it for the actual algorithm!
        # Print statistics, save files
        lossnum = loss.data[0].item()   # Saved loss is the actual learning loss (MSE)
        #to = target.cpu().numpy(); yo = torch.stack(out).data.cpu().numpy(); z = (yo - to) ** 2; lossnum = np.mean(z[:])  # Saved loss is the error rate
        
        ll.append(loss.item())

        total_loss  += lossnum
        if (numiter+1) % print_every == 0:
            sz = min(50, len(ll))
            box = np.ones(sz)/sz
            lll = torch.Tensor(ll[::10])
            lc = np.convolve(lll, box, mode='valid')

            vis.line(Y=lc, X=torch.arange(lc.shape[0]),
                win=str(exp_id) + 'learning_curve',
                opts=dict(title='Training Curve', width=400, height=300))


            pca = sklearn.decomposition.PCA(n_components=2)
            YY = torch.cat(ys).detach().cpu().numpy()
            pca.fit(YY)
            XX = pca.transform(YY)

            vis.scatter(X=XX,
                win='pca' + str(exp_id) + str((numiter+1) % 3),
                opts=dict(
                    markercolor=(np.arange(T)/T * 255).astype(np.int),
                    title='pca' + str(exp_id) + str((numiter+1) % 3)
                ))

            vis.scatter(X=np.concatenate([XX,  (np.arange(T)[:,None] - T/2) /T], axis=-1),
                win='pca3d' + str(exp_id) + str((numiter+1) % 3),
                opts=dict(
                    markercolor=(np.arange(T)/T * 255).astype(np.int),
                    title='pca' + str(exp_id) + str((numiter+1) % 3)
                ))

            # vis.line(X=torch.cat([inputs.squeeze(-1), inputs.squeeze(-1)], dim=-1), Y=torch.cat([target.squeeze(-1), torch.cat(out)], dim=-1), win=str(exp_id) + str((numiter+1) % 3), opts=dict(title=str(exp_id) + str((numiter+1) % 3)))

            # XX = torch.cat([torch.cat([inputs.squeeze(-1), inputs.squeeze(-1)], dim=0), torch.cat([target.squeeze(-1), torch.cat(out)], dim=0)], dim=1)
            # YY = torch.cat([torch.ones(target.squeeze(-1).shape), torch.ones(target.squeeze(-1).shape) + 1], dim=0)
            # vis.scatter(Y=YY, X=XX, win='scatter'+str(exp_id) + str((numiter+1) % 3), opts=dict(title=str(exp_id) + str((numiter+1) % 3)))
            if e > n_pretrain:
                vis.images(torch.cat(out),padding=2, nrow=20, win='vid'+str(exp_id) + str((numiter+1) % 3), opts=dict(title=str(exp_id) + str((numiter+1) % 3), 
                                    height=50*(target.shape[0]//20 + 1), width=50*23))

                vis.images(target.unsqueeze(1), nrow=20, win='vidtarg'+str(exp_id) + str((numiter+1) % 3), opts=dict(title=str(exp_id) + str((numiter+1) % 3), 
                                    height=50*(target.shape[0]//20 + 1), width=50*23))

                import pdb; pdb.set_trace()
            else:
                vis.images(torch.cat([o[0] for o in out]).unsqueeze(1), 
                    nrow=20, win='vid'+str(exp_id) + str((numiter+1) % 3),
                    opts=dict(title=str(exp_id) + str((numiter+1) % 3),
                    height=50*(target.shape[0]//20 + 1), width=50*21))
                vis.images(target[:, 0].unsqueeze(1), nrow=20, win='vidtarg'+str(exp_id) + str((numiter+1) % 3), 
                    opts=dict(title=str(exp_id) + str((numiter+1) % 3), 
                    height=50*(target.shape[0]//20 + 1), width=50*21))

            # import pdb; pdb.set_trace()

            print((e, numiter, "===="))
            # print(target.cpu().numpy()[-10:].squeeze())   # Target pattern to be reconstructed
            # print(inputs.cpu().numpy()[numstep][0][-10:])  # Last input contains the degraded pattern fed to the network at test time
            # print(torch.stack(out).data.cpu().numpy()[-10:].squeeze())   # Final output of the network

            previoustime = nowtime
            nowtime = time.time()
            print("Time spent on last", print_every, "iters: ", nowtime - previoustime)
            total_loss /= print_every
            all_losses.append(total_loss)
            print("Mean loss over last", print_every, "iters:", total_loss)

            if e > n_pretrain:
                print("Eta: ", net.eta.item(), "mean Alpha", net.alpha.mean().item())
                print("Mean Eta: ", sum([e.abs().mean() for e in etas]).item() / len(etas), "Eta sparsity: ", sum([(e>0.1).float().mean() for e in etas]).item() / len(etas))
                # print(sum([e.abs()/ e.sum() for e in etas]) / len(etas))

            # with open('output_simple_'+str(RNGSEED)+'.dat', 'wb') as fo:
            #     pickle.dump(net.w.data.cpu().numpy(), fo)
            #     pickle.dump(net.alpha.data.cpu().numpy(), fo)
            #     pickle.dump(y.data.cpu().numpy(), fo)  # The final y for this episode
            #     pickle.dump(all_losses, fo)
            # with open('loss_simple_'+str(RNGSEED)+'.txt', 'w') as fo:
            #     for item in all_losses:
            #         fo.write("%s\n" % item)
            total_loss = 0



