#!/usr/bin/env python

import os
import torch
import numpy as np
import sys
from sklearn import linear_model
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import collections
from src.torch_hsic import *

def basic_optimizer(model, train_loader, lr=0.001, max_loop=2000):
	if torch.cuda.is_available(): device = 'cuda'
	#loss = nn.CrossEntropyLoss()

	optimizer = torch.optim.Adam(model.parameters(), lr=lr)	
	scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau( optimizer, factor=0.5, min_lr=1e-10, patience=50, verbose=False)

	σ = Variable(torch.rand(1).to(device), requires_grad=True)
	σ = 0.01
	for epoch in range(max_loop):

		loss_list = []	
		for (i, data) in enumerate(train_loader):
			[X, Y] = data
			X = X.to(device)
			Y = Y.to(device)
		
			#Yₒ = OneHotEncoder(categories='auto', sparse=False).fit_transform(np.atleast_2d(Y).T)
			#Yₒ= torch.tensor(Yₒ).to(device)

			optimizer.zero_grad()			
			xout = model(X)

			output = model.CE_loss(xout, Y, epoch, i, train_loader.batch_size)
			#output, nHSIC = torch_hsic(xout, Yₒ, σ)
			#print('%d, %d, %.3f, %.3f, %.3f'%(epoch, i, output.item(), nHSIC, σ))


			output.backward()
			optimizer.step()
			#σ.data -= lr*σ.grad.data

			loss_list.append(output.item())

		loss_avg = np.array(loss_list).mean()
		scheduler.step(loss_avg)
		
		if os.path.exists('exit.txt'): break





