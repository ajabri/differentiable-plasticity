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
parser.add_argument("--name", help="network type ('plastic' or 'nonplastic')", default='auto')
parser.add_argument("--reload_vae", help="", default='')
parser.add_argument("--reload", help="", default='')
parser.add_argument("--dataset", help="", default='balls')
parser.add_argument("--T", type=int, help="number of time steps between each pattern presentation (with zero input)", default=20)
parser.add_argument("--n-adapt", type=float, default=0.5, help="number of time steps between each pattern presentation (with zero input)")
parser.add_argument("--nhid", type=int, default=50, help="number of time steps between each pattern presentation (with zero input)")
parser.add_argument("--rnn-type", type=str, default='GRU', help="GRU | RNN")

args = parser.parse_args(); argvars = vars(args); argdict =  { k : argvars[k] for k in argvars if argvars[k] != None }
params.update(argdict)

if 'auto' in params['name']:
    name = ''
    for k in ['type', 'dataset', 'T', 'n_adapt', 'nhid', 'rnn_type']:
        name += "%s[%s]" % (k, params[k])
    params['name'] = params['name'].replace('auto', name)

import visdom
exp_id = int(time.time()/1000)
vis = visdom.Visdom(port=8095, env=str(params['name']) + ' (%s)' % exp_id)
import time


RNGSEED = 0             # Initial random seed - can be modified by passing a number as command-line argument

if len(sys.argv) == 2:
    RNGSEED = int(sys.argv[1])
    print("Setting RNGSEED to "+str(RNGSEED))
np.set_printoptions(precision=3)
np.random.seed(RNGSEED); random.seed(RNGSEED); torch.manual_seed(RNGSEED)


#ttype = torch.FloatTensor;         # For CPU
ttype = torch.cuda.FloatTensor;     # For GPU 

bce_loss = nn.BCELoss().type(ttype)


from plastic_rnn import GRU, RNN


class PlasticPixelPredictor(nn.Module):
    def __init__(self, nin, nhid, nout, hebb='h,z,r', ln=True):
        super(PlasticPixelPredictor, self).__init__()

        if params['rnn_type'] == 'GRU':
            self.rnn = GRU(nin, nhid, hebb=set(hebb.split(',')), ln=True)
        else:
            self.rnn = RNN(nin, nhid, hebb=params['type']=='plastic', ln=True)

        self.out = nn.Linear(nhid, nout)
        
        self.enc = vae.encoder()
        self.dec = vae.decoder()

    def forward(self, x, hid, hebb=None):
        # x_orig = x
        # B, T = x.shape[:2]
        # if len(x.shape) > 3:
        #     B, T = x.shape[:2]
        #     x = x.view(B * T, *x.shape[2:])
        
        x = self.enc(x)
        # if len(x_orig.shape) > 3:
        #     x = x.view(B, T, *x.shape[1:])
        
        # import pdb; pdb.set_trace()

        hid, hebb = self.rnn(x, hid, hebb)
        y = self.out(hid)

        out = self.dec(y)
        # HACK
        self.eta_hat = torch.zeros(1) if not hasattr(self.rnn.h.hid_lin, 'eta_hat') else self.rnn.h.hid_lin.eta_hat
        self.lamda = torch.zeros(1) if not hasattr(self.rnn.h.hid_lin, 'lamda') else self.rnn.h.hid_lin.lamda

        return hid, hebb, out

    def initialZeroState(self):
        # Return an initialized, all-zero hidden state
        return self.rnn.initialZeroState()

    def initialZeroHebb(self):
        # Return an initialized, all-zero Hebbian trace
        return self.rnn.initialZeroHebb()

# net = NETWORK()

net = PlasticPixelPredictor(nin=300, nhid=params['nhid'], nout=300, hebb=('h'), ln=True).cuda()

# import pdb; pdb.setTrace()

##############

if os.path.exists(params['reload_vae']) or os.path.exists(params['reload']):
    if os.path.exists(params['reload']):
        reloaded = torch.load(params['reload'])
        net.load_state_dict(reloaded)

    if os.path.exists(params['reload_vae']):
        reloaded = list(torch.load(params['reload_vae']))

        net.enc, net.dec = reloaded
        net.enc, net.dec = net.enc.cuda(), net.dec.cuda()

    optimizer = torch.optim.Adam(list(net.rnn.parameters()) + list(net.out.parameters()), lr=params['lr'])
    # optimizer = torch.optim.Adam([net.alpha, net.eta,
    #     net.pred_eta, net.pred_eta_b, #,
    #     net.b1, net.b0, 
    #     net.w0, net.w1, net.w], lr=params['lr'])

    # import pdb; pdb.setTrace()
else:
    optimizer = torch.optim.Adam(net.parameters(), lr=params['lr'])

    # optimizer = torch.optim.Adam([net.w, net.alpha, net.eta, net.pred_eta,
    #     net.b1, net.b0, net.pred_eta_b,# net.ln
    #     *list(net.enc.parameters()), *list(net.dec.parameters()),
    #     net.w0, net.w1], lr=params['lr'])

# optimizer = torch.optim.SGD([net.w, net.alpha, net.eta,
#     net.b1, net.b0, net.pred_eta, net.pred_eta_b,
#     net.w0, net.w1],
#     lr=0.01)

# net = net.cuda()

total_loss = 0.0; all_losses = []
print_every = 100
nowtime = time.time()


import scipy
import scipy.io

if params['dataset'] == 'mnist':
    mm1 = np.load('/data/ajabri/moving_mnist/mnistTest_seq_28.npy')

    hw = 16
    mm = np.ndarray((*mm1.shape[:-2], hw, hw))
    for t in range(mm1.shape[0]):
        for n in range(mm1.shape[1]):
            mm[t][n] = scipy.misc.imresize(mm1[t][n], (hw, hw))

elif params['dataset'] == 'balls':
    mm1 = scipy.io.loadmat('./bouncing_balls_training_data_2.mat')
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
if os.path.exists(params['reload_vae']) or os.path.exists(params['reload'] or True):
    n_pretrain = -1

ll = []

for e in range(100):
    torch.save(net.state_dict(), '/data/ajabri/moving_mnist/checkpoints/%s_%s.pth' % (params['name'], exp_id))

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
        optimizer.zero_grad()
        ys = []

        if e > n_pretrain:
            if not saved and not os.path.exists(params['reload_vae']):
                torch.save( {net.dec, net.enc}, '/data/ajabri/moving_mnist/vae.pth')
                saved = True

            # inputs = mm[:, numiter].type(ttype)
            # target = mm[:, numiter+1].type(ttype)

            idx = np.random.choice(mm.shape[1] - 1, 30)
            inputs = mm[:, idx].type(ttype).transpose(0, 1)
            target = mm[:, idx+1].type(ttype).transpose(0, 1)

            # inputs = torch.stack([inputs, inputs])
            # target = torch.stack([target, target])

            # import pdb; pdb.set_trace()

            hebb = net.initialZeroHebb()
            y = net.initialZeroState()
            for numstep in range(T):
                if numstep > T // 2:
                    y, hebb, pred = net(Variable(pred, requires_grad=False), y, hebb)
                else:
                    y, hebb, pred = net(Variable(inputs[:, numstep].unsqueeze(1), requires_grad=False), y, hebb)

                out.append(pred)
                etas.append(net.eta_hat)
                ys.append(y)

        else:
            idxs = torch.randperm(mm.shape[1]-1)[:100]
            inputs = mm[:, idxs].type(ttype)
            target = mm[:, idxs+1].type(ttype)

            for numstep in range(T):
                out.append(net.dec(net.enc(Variable(inputs[numstep].unsqueeze(1), requires_grad=False))))

        loss = bce_loss(torch.stack(out).squeeze().transpose(0,1)[:], Variable(target, requires_grad=False)[:])
        ll.append(loss)
        # Apply backpropagation to adapt basic weights and plasticity coefficients
        loss.backward()
        # print
        (torch.nn.utils.clip_grad_norm_(optimizer.param_groups[0]['params'], 10))
        optimizer.step()


        # That's it for the actual algorithm!
        # Print statistics, save files
        lossnum = loss.item()   # Saved loss is the actual learning loss (MSE)
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
            YY = torch.stack(ys)[:, 0].detach().cpu().numpy()
            pca.fit(YY)
            XX = pca.transform(YY)

            vis.scatter(X=XX,
                win='pca' + str(exp_id) + str((numiter+1) % 3) + str(T==T),
                opts=dict(
                    markercolor=(np.arange(T)/T * 255).astype(np.int),
                    title='pca' + str(exp_id) + str((numiter+1) % 3) + str(T==T)
                ))

            vis.scatter(X=np.concatenate([XX,  (np.arange(T)[:,None] - T/2) /T], axis=-1),
                win='pca3d' + str(exp_id) + str((numiter+1) % 3) + str(T==T),
                opts=dict(
                    markercolor=(np.arange(T)/T * 255).astype(np.int),
                    title='pca' + str(exp_id) + str((numiter+1) % 3) + str(T==T)
                ))

            # vis.line(X=torch.cat([inputs.squeeze(-1), inputs.squeeze(-1)], dim=-1), Y=torch.cat([target.squeeze(-1), torch.cat(out)], dim=-1), win=str(exp_id) + str((numiter+1) % 3), opts=dict(title=str(exp_id) + str((numiter+1) % 3)))

            # XX = torch.cat([torch.cat([inputs.squeeze(-1), inputs.squeeze(-1)], dim=0), torch.cat([target.squeeze(-1), torch.cat(out)], dim=0)], dim=1)
            # YY = torch.cat([torch.ones(target.squeeze(-1).shape), torch.ones(target.squeeze(-1).shape) + 1], dim=0)
            # vis.scatter(Y=YY, X=XX, win='scatter'+str(exp_id) + str((numiter+1) % 3), opts=dict(title=str(exp_id) + str((numiter+1) % 3)))
            if e > n_pretrain:
                vis.images(torch.stack(out).squeeze().transpose(0,1)[0].unsqueeze(1), nrow=20, win='vid'+str(exp_id) + str((numiter+1) % 3), opts=dict(title=str(exp_id) + str((numiter+1) % 3), 
                                    height=50*(target.shape[0]//20 + 1), width=50*20))

                vis.images(target[0].unsqueeze(1), nrow=20, win='vidtarg'+str(exp_id) + str((numiter+1) % 3), opts=dict(title=str(exp_id) + str((numiter+1) % 3), 
                                    height=50*(target.shape[0]//20 + 1), width=50*20))

            else:
                vis.images(torch.cat([o[0] for o in out]).unsqueeze(1), 
                    nrow=20, win='vid'+str(exp_id) + str((numiter+1) % 3),
                    opts=dict(title=str(exp_id) + str((numiter+1) % 3),
                    height=50*(target.shape[0]//20 + 1), width=50*20))
                vis.images(target[:, 0].unsqueeze(1), nrow=20, win='vidtarg'+str(exp_id) + str((numiter+1) % 3), 
                    opts=dict(title=str(exp_id) + str((numiter+1) % 3), 
                    height=50*(target.shape[0]//20 + 1), width=50*20))

            # import pdb; pdb.setTrace()

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
                # print("Eta: ", net.eta.item(), "mean Alpha", net.alpha.mean().item())
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



