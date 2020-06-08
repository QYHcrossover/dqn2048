'''
Author:qinyuheng
Mail:qinyuheng@hhu.edu.cn
Github:QYHcrossover
Date:2020/06/06
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Network(nn.Module):
	def __init__(self,depth1=128,depth2=128,input_units=16,hidden_units=256,output_units=4):
		super(Network,self).__init__()
		self.conv11 = nn.Conv2d(in_channels=input_units,out_channels=depth1,kernel_size=(1,2))
		self.conv21 = nn.Conv2d(in_channels=input_units,out_channels=depth1,kernel_size=(2,1))
		self.conv12 = nn.Conv2d(in_channels=depth1,out_channels=depth2,kernel_size=(1,2))
		self.conv22 = nn.Conv2d(in_channels=depth1,out_channels=depth2,kernel_size=(2,1))
		expand_size = 2*4*depth2*2 + 3*3*depth2*2 + 4*3*depth1*2
		self.fc1 = nn.Linear(in_features=expand_size,out_features=hidden_units)
		self.fc2 = nn.Linear(in_features=hidden_units,out_features=output_units)

	def forward(self,x):
		# print("x.size:{}".format(x.shape))
		batch_size = x.size(0)
		#第一层卷积
		conv1 = self.conv11(x)
		conv1 = F.relu(conv1, inplace=True)
		# print("conv1.size:{}".format(conv1.shape))
		# print("x.size:{}".format(x.shape))
		conv2 = self.conv21(x)
		conv2 = F.relu(conv2, inplace=True)
		#第二层卷积
		conv11 = self.conv12(conv1)
		conv11 = F.relu(conv11, inplace=True)
		conv12 = self.conv22(conv1)
		conv12 = F.relu(conv12, inplace=True)
		conv21 = self.conv12(conv2)
		conv21 = F.relu(conv21, inplace=True)
		conv22 = self.conv22(conv2)
		conv22 = F.relu(conv22, inplace=True)
		#全连接
		x = torch.cat([conv1.view(batch_size,-1),conv2.view(batch_size,-1),
			conv11.view(batch_size,-1),conv12.view(batch_size,-1),conv21.view(batch_size,-1),conv22.view(batch_size,-1)],dim=1)
		x = F.relu(self.fc1(x),inplace=True)
		x = self.fc2(x)
		return x

	@staticmethod
	def initWeights(m):
		if type(m) == nn.Conv2d or type(m) == nn.Linear:
			nn.init.uniform_(m.weight, -0.01, 0.01)
			m.bias.data.fill_(0.01)

if __name__ == "__main__":
	array = np.random.randn(512,16,4,4)
	tensor = torch.from_numpy(array).float().to("cuda")
	nk = Network()
	nk.cuda()
	nk.apply(Network.initWeights)
	print(nk.forward(tensor))