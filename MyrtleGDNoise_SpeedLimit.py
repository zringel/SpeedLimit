import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.autograd import Function
from torch.nn.parameter import Parameter
from torch import autograd 

import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.utils import make_grid
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, TensorDataset


import numpy as np
from numpy import linalg as LA
import random

# from mpltools import annotation

from scipy.io import savemat, whosmat
from scipy.optimize import fsolve
import time
import os
import sys 
from tempfile import TemporaryFile

import copy
import pathlib
import scipy.signal

import io
import signal
import noisy_sgd 

import argparse

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def calc_run_time(start, end):
    seconds = int(end - start)
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    run_time = '{:d}:{:02d}:{:02d}'.format(h, m, s)
    print('run_time =', run_time)
    return run_time

def signal_handler(sig, frame):
        global stop_me
        print('stopping')
        stop_me = True
    

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------    
    
parser = argparse.ArgumentParser(description="MyrtleGDNoise_SpeedLimit.")

parser.add_argument('--lr0', help="lr0 ",type=float, default=3e-08)
parser.add_argument('--lr_fac', help="lr_fac - factor dividing the highest spike free learning rate ",type=float, default=2.0)
parser.add_argument('--num_channels', help="num_channels - number of channels", type=int, default=128)
parser.add_argument('--n_train', help="n_train - number training pts", type=int, default=100)
parser.add_argument('--train_seed', help="train_seed - random seed used for noisy training", type=int, default=111)
parser.add_argument('--folder', help="folder to save results", default='./tmp/')

c = parser.parse_args()
torch.set_num_threads(4)


stop_me = False
signal.signal(signal.SIGINT, signal_handler)
# Collect events until released

activation_type = 'relu' # 'erf', 'relu'

train_protocol = 'Langevin' # 'SGD', 'Langevin'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = 'cpu'
max_epochs = 200000 # min((10 * t_eq + t_eq), 1000000) 
start_rec_nets = 0

num_channels = c.num_channels # np.int(sys.argv[2])
kernel_size = 3

train_seed_type = 'consistent'  # 'consistent', 'random'
train_seed = c.train_seed
data_seed = 111

num_classes = 10 
print_every = 100
save_nets_every = 1


half = False
square_loss = True
dtype=torch.float32
loss_fn = "mse"
zca_str, aug_str, half_str = '', '', ''

    
if train_protocol == 'Langevin':     
    # GD+noise hyper-params  
    sample = 'first' # 'first', 'random'
    lr_fac = c.lr_fac  # np.float(sys.argv[1]) # This will the factor dividing the highest spike free learning rate as found by the algorithm 
    n_train = c.n_train # np.int(sys.argv[3])
    # lr0 = (0.001/np.float(n_train))*np.min([(128./num_channels), 1]) 
    # lr0 = (0.0005/np.float(n_train))*np.min([(128./num_channels), 1]) 
    lr0 = c.lr0
    n_test = n_train
    sample_seed = 123 # for sampling the training and test data
    batch_size = n_train # full-batch
    sigma2 = 1e-10 / 2.0
    temperature = 2.0*sigma2  # notice factor of 2.0
    prefactor = 1/2.0
    wd_input = prefactor * temperature * 3.0 * kernel_size**2
    wd_hidden = prefactor * temperature * num_channels * kernel_size**2
    wd_output = prefactor * temperature * num_channels
    mse_reduction = 'sum'    # use SE loss which is the sum, not the average

print('max_epochs = {} | save_nets_every = {}'.format(max_epochs, save_nets_every) )    
print('wd_input = {} wd_hidden = {} wd_output = {}'.format(wd_input, wd_hidden, wd_output))



# folder and file names
exp_folder = c.folder
exp_name = 'myrtle5_TrainBugFixed__C={}__n_train={}__train_seed={}__lr_fac={}__sigma2={}__activation={}__lr0={}'.format(
                    num_channels, n_train, train_seed, lr_fac, sigma2, activation_type,lr0)        
if not os.path.exists(exp_folder):
    os.mkdir(exp_folder)
results_file_name = r'{}/{}.npz'.format(exp_folder, exp_name)    

    

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ 


transform = transforms.Compose(
    [transforms.ToTensor()])

train_data = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
test_data = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)

if train_protocol == 'SGD':
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                          shuffle=False, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

elif train_protocol == 'Langevin':
    # take a subsample
    if sample == 'random':
        random.seed(sample_seed)
        train_inds = sorted(random.sample(range(len(train_data)), n_train))
        test_inds = sorted(random.sample(range(len(test_data)), n_test))
    elif sample == 'first':
        train_inds = list(range(n_train))
        test_inds = list(range(n_test))

    train_data = torch.utils.data.Subset(train_data, train_inds)
    test_data = torch.utils.data.Subset(test_data, test_inds)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=2)

class Myrtle_5_CNN(nn.Module):
    def __init__(self):
        super().__init__() 
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=num_channels, kernel_size=kernel_size, padding=(1,1),bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=kernel_size, padding=(1,1),bias=False)
        self.conv3 = nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=kernel_size, padding=(1,1),bias=False)
        self.conv4 = nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=kernel_size, padding=(1,1),bias=False)
        self.linear = nn.Linear(in_features=num_channels, out_features=10, bias=False)

    def forward(self, x):
        x = activation(self.conv1(x))
        x = self.pool(activation(self.conv2(x)))
        x = self.pool(activation(self.conv3(x)))
        x = self.pool(activation(self.conv4(x)))        
        x = self.pool(x)                
        x = self.pool(x)    
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = self.linear(x)

        return x
    

best_acc = 0.0

if train_seed_type == 'random':
    torch.seed()      # make sure DNN initialization is different every time 
    np.random.seed()
elif train_seed_type == 'consistent':
    torch.manual_seed(train_seed)
    np.random.seed(train_seed)    

# build net    
net = Myrtle_5_CNN()

# multiply weights by sqrt(6) to get the proper initialization - one that matches our P_0
for i in [1,3,4,5,-1]:
    list(net.modules())[i].weight = nn.Parameter(list(net.modules())[i].weight*np.sqrt(6))

net.to(device)

criterion = nn.MSELoss(reduction=mse_reduction)


print("n_train | lr_fac | num_channels = ", n_train, lr_fac, num_channels)

if activation_type == 'relu':
    activation = torch.nn.ReLU()
    lr_epoch_list = [(200000, lr0/2.),
                 (600000, 2.7343749999999997e-05),
                 (700000, 1.9140624999999996e-05),
                 (800000, 1.3398437499999996e-05),
                 (900000, 9.378906249999996e-06),
                 (1300000, 6.565234374999997e-06),
                 (1800000, 4.595664062499998e-06),
                 (2200000, 3.2169648437499985e-06),
                 (2500000, 2.251875390624999e-06),
                 (2800000, 6.755626171874996e-07)] # 2.251875390624999e-06 * 0.3
elif activation_type == 'erf':
    activation = torch.special.erf
    lr_div = 2.0
    lr0 = lr0 / lr_div
    lr_epoch_list = [(100, 3.90625e-05 / lr_div),
                 (60000, 2.7343749999999997e-05 / lr_div),
                 (70000, 1.9140624999999996e-05 / lr_div),
                 (80000, 1.3398437499999996e-05 / lr_div),
                 (90000, 9.378906249999996e-06 / lr_div),
                 (130000, 6.565234374999997e-06 / lr_div),
                 (180000, 4.595664062499998e-06 / lr_div),
                 (220000, 3.2169648437499985e-06 / lr_div),
                 (250000, 2.251875390624999e-06 / lr_div),
                 (280000, 6.755626171874996e-07 / lr_div)] # 2.251875390624999e-06 * 0.3

if train_protocol == 'Langevin':     
    optimizer = noisy_sgd.LangevinMyrtle5(net, lr0, wd_input, wd_hidden, wd_output, temperature)    

# main training loop
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
print(torch.cuda.get_arch_list())
print(torch.version.cuda)
losses_train, losses_test = [], []
accs_train, accs_test = [], []
outputs_train, outputs_test = [], []
nets = []
curLr = lr0

start = time.time()
lr_ind = 0

for i, data in enumerate(train_loader, 0): # for Langevin training we use full-batch
    inputs, labels = data  
    inputs, labels = inputs.to(device), labels.to(device)
total_grads = []
for epoch in range(max_epochs+1):
    
        
    if (epoch % save_nets_every == 0):
        print('snapshot of model kept')
        best_model_wts = copy.deepcopy(net.state_dict())
        nets += [best_model_wts]
        if epoch == 101:
            save_nets_every=100

    net.train()
    running_loss = 0.0
    train_correct = 0
    train_total = 0
    
    optimizer.zero_grad()
    outputs = net(inputs).view(-1, num_classes)        
    _, predicted = torch.max(outputs.data, 1)
    
    train_total += labels.size(0)
    train_correct += (predicted == labels).sum().item() # for the accuracy, labels are integers
    if square_loss:
        # for the loss, labels are one-hot encoded using scatter_()
        labels_1hot = torch.FloatTensor(outputs.size()).zero_().scatter_(1, labels.detach().cpu().reshape(outputs.size()[0], 1), 1).to(dtype=dtype, device=device)
    loss = criterion(outputs, labels_1hot)
    loss.backward()
        
    total_grads += [(optimizer.step(),curLr)]
    
    # determine lr according to a fixed scheduler
    if epoch == lr_epoch_list[lr_ind][0]:
        curLr = lr_epoch_list[lr_ind][1]
        if lr_ind < len(lr_epoch_list) - 1: 
            lr_ind += 1
        optimizer = noisy_sgd.LangevinMyrtle5(net, curLr, wd_input, wd_hidden, wd_output, temperature)
    
    running_loss += loss.item()
        
    if train_protocol == 'SGD':
        lr_scheduler.step()        
        
    if epoch % print_every == 0:
        outputs_train += [outputs]
        
        net.eval()
        test_correct, test_total, test_loss = 0, 0, 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                if half:
                    inputs, labels = inputs.to(device).half(), labels.to(device)
                else:
                    inputs, labels = inputs.to(device), labels.to(device)
                outputs = net(inputs).view(-1, num_classes)
                outputs_test += [outputs]
                
                _, predicted = torch.max(outputs.data, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item() # for the accuracy, labels are integers
                if square_loss:
                    # for the loss, labels are one-hot encoded using scatter_()
                    labels_1hot = torch.FloatTensor(outputs.size()).zero_().scatter_(1, labels.detach().cpu().reshape(outputs.size()[0], 1), 1).to(dtype=dtype, device=device)
                loss = criterion(outputs, labels_1hot)
                test_loss += loss.item()
                
        trainacc = train_correct/train_total*100
        testacc = test_correct/test_total*100
        
        print('Epoch: '+ str(epoch),'| TrLoss: ' + str(running_loss),"| TrAcc: " + str(trainacc),'| TeLoss: ' + str(test_loss), "| TeAcc: " + str(testacc), '| curLr:' + str(curLr))

        losses_train += [running_loss]
        losses_test += [test_loss]
        accs_train += [trainacc]
        accs_test += [testacc]        
    
        for i, data in enumerate(train_loader, 0): # for Langevin training we use full-batch
            inputs, labels = data  
            inputs, labels = inputs.to(device), labels.to(device)
    if stop_me==True:
        print('terminating early')
        break

        

end = time.time()        
run_time = calc_run_time(start, end)   


# save data
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
data_dict = {
            'epoch': epoch,
            'nets': nets,
            'outputs_train': outputs_train,
            'outputs_test': outputs_test,
            'lr_epoch_list': lr_epoch_list,
            'losses_train': losses_train,
            'losses_test': losses_test,
            'accs_train': accs_train,
            'accs_test': accs_test,
            'run_time': run_time,
            'num_channels': num_channels,
            'n_train': n_train,
            'train_seed': train_seed,
            'lr_fac': lr_fac,
            'sigma2': sigma2,
            'activation_type': activation_type,
            'grad2':total_grads
            }
torch.save(data_dict, results_file_name)       
