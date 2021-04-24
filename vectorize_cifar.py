#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
Created on Wed Mar 24 15:42:53 2021

@author: tong
"""
import torch
from tqdm import tqdm
import pickle
from src.resnet import *
from torchvision import transforms, utils, datasets
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from src.basic_optimizer import * 
from sklearn.svm import SVC
from numpy import moveaxis

torch.set_printoptions(edgeitems=3)
torch.set_printoptions(sci_mode=False)
torch.set_printoptions(precision=2)
torch.set_printoptions(linewidth=400)


def get_cifar10_data(data_folder_path, batch_size=64):
	""" cifar10 data
	Args:
		train_batch_size(int): training batch size 
		test_batch_size(int): test batch size
	Returns:
		(torch.utils.data.DataLoader): train loader 
		(torch.utils.data.DataLoader): test loader
	"""
	transform_train = transforms.Compose([

		transforms.RandomCrop(32, padding=4),
		transforms.RandomHorizontalFlip(),
		transforms.ToTensor(),
		transforms.Normalize((0.4913, 0.4821, 0.4465), (0.2470, 0.2434, 0.2615)),

	])
	transform_test = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize((0.4913, 0.4821, 0.4465), (0.2470, 0.2434, 0.2615)),
	])

	train_data = datasets.CIFAR10(data_folder_path, train=True, download=True, transform=transform_train)
	test_data  = datasets.CIFAR10(data_folder_path, train=False, download=True, transform=transform_test) 
	
	kwargs = {'num_workers': 4, 'pin_memory': True}
	train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, **kwargs)
	test_loader  = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, **kwargs)

	return train_loader, test_loader




#	Downloading the data
#trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
#trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)
#
#testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
#testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)
#
#classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


train_loader, test_loader = get_cifar10_data('./', batch_size=64)	
#model = pickle.load( open( "model.pk", "rb" ) )
model = ResNet18(num_classes=10)
model.to('cuda')
basic_optimizer(model, train_loader, max_loop=20) #200


# --------
model.eval()
Xsave = np.empty((0,10))
Ysave = np.empty((0))
avg_Acc = []
for (i, data) in enumerate(train_loader):
	[X, Y] = data
	X = X.to('cuda')
	Y = Y.to('cuda')

	Xout = model(X)

	SM = torch.nn.functional.softmax(Xout, dim=1)
	aa, ŷ = torch.max(SM, dim=1)
	Acc = torch.sum(ŷ == Y)/Xout.shape[0]
	avg_Acc.append(Acc.cpu().item())

	Xout = Xout.data.cpu().numpy()
	Xsave = np.vstack((Xsave, Xout))
	Ysave = np.hstack((Ysave, Y.cpu().numpy()))


print('Averag Accuracy Training : %.3f'%np.mean(avg_Acc))

pickle.dump( model, open( "model.pk", "wb" ) )
np.savetxt('cifar10.csv', Xsave, delimiter=',', fmt='%.4f') 
np.savetxt('cifar10_label.csv', Ysave, delimiter=',', fmt='%d') 


Xsave = np.empty((0,10))
Ysave = np.empty((0))
avg_Acc = []
for (i, data) in enumerate(test_loader):
	[X, Y] = data
	X = X.to('cuda')
	Y = Y.to('cuda')

	Xout = model(X)

	SM = torch.nn.functional.softmax(Xout, dim=1)
	aa, ŷ = torch.max(SM, dim=1)
	Acc = torch.sum(ŷ == Y)/Xout.shape[0]
	avg_Acc.append(Acc.cpu().item())

	Xout = Xout.data.cpu().numpy()
	Xsave = np.vstack((Xsave, Xout))
	Ysave = np.hstack((Ysave, Y.cpu().numpy()))

print('Averag Accuracy Test : %.3f'%np.mean(avg_Acc))
np.savetxt('cifar10_test.csv', Xsave, delimiter=',', fmt='%.4f') 
np.savetxt('cifar10_label_test.csv', Ysave, delimiter=',', fmt='%d') 



## Try to classify the data
#from numpy import genfromtxt
#
#X = genfromtxt('cifar10.csv', delimiter=',')
#Y = genfromtxt('cifar10_label.csv', delimiter=',')
#Xⲧ = genfromtxt('cifar10_test.csv', delimiter=',')
#Yⲧ = genfromtxt('cifar10_label_test.csv', delimiter=',')
#
#svm = SGDClassifier(max_iter=5000, tol=1e-3, verbose=False)
#svm.fit(X, Y)
#Ŷ = svm.predict(X)
#Ŷⲧ = svm.predict(Xⲧ)
#
#ᘔ = accuracy_score(Y, Ŷ)
#ᘔⲧ = accuracy_score(Yⲧ, Ŷⲧ)
#
#print(ᘔ, ᘔⲧ)
#import pdb; pdb.set_trace()

