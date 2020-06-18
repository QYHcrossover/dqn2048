'''
Author:qinyuheng
Mail:qinyuheng@hhu.edu.cn
Github:QYHcrossover
Date:2020/06/06
'''
from network import Network
from game2048 import Game2048
from collections import deque
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Agent:
	def __init__(
		self,
		batch_size = 32,
		max_eposide = 200000,
		save_per_iter = 20,
		memory_size = 6000,
		):
		#dqn基础参数
		self.epsilon = 0.9 #初始探索率较高
		self.reward_decay = 1 #衰减因子
		self.memory_size = memory_size
		self.memory = deque(maxlen=memory_size) #经验池
		#network相关
		self.batch_size = batch_size #batch_size
		self.network = Network().cuda() #决策网络
		self.network.apply(Network.initWeights)
		# self.optimizer = torch.optim.Adam(self.network.parameters(), lr=1e-4) #todo,学习率衰减
		self.optimizer = torch.optim.RMSprop(self.network.parameters(),lr=0.0005) #todo,学习率衰减
		self.scheduler = torch.optim.lr_scheduler.StepLR(optimizer=self.optimizer,step_size=1000,gamma=0.9)
		self.loss_fn = nn.MSELoss()
		#训练控制相关
		self.max_eposide = max_eposide #最大eposide大小
		self.save_per_iter = save_per_iter #n，每训练n步保存一次
		self.eposide = 0 #当前eposide数
		self.train_steps = 0 #当前训练了多少步
		self.total_steps = 0 #总共走了多少步
		self.record = [] #保存游戏数据
		#最优参数
		self.max_num = 0
		self.max_score = 0

	def transform_state(self,state):
		assert type(state) in (tuple,np.ndarray)
		def transform_one(state):
			new = np.zeros(shape=(1,16,4,4),dtype=np.float32)
			for i in range(4):
				for j in range(4):
					if(state[i][j]==0):
						new[0][0][i][j] = 1.0
					else:
						loc = int(math.log(state[i][j],2))
						new[0][loc][i][j] = 1.0
			return new
		if type(state) is np.ndarray:
			return transform_one(state)
		else:
			return np.concatenate([transform_one(s) for s in state],axis=0)

	def chooseAction(self,state,e_greedy=True):
		#esilon衰减
		if e_greedy:
			if((self.eposide>10000) or (self.epsilon>0.1 and self.total_steps % 2500 == 0)):
				self.epsilon /= 1.005
			if np.random.rand() < self.epsilon:
				return np.random.rand(4)
		state = self.transform_state(state)
		feed = torch.from_numpy(state).to("cuda")
		result = self.network.forward(feed)
		return result.cpu().detach().numpy()[0]

	def update_randomsample(self):
		#从memeory中采样batch_size个
		sample_batch = random.sample(self.memory,self.batch_size)
		state_batch,action_batch,nextstate_batch,reward_batch,isover_batch = zip(*sample_batch)
		feed_batch = torch.from_numpy(self.transform_state(state_batch)).to("cuda")

		#计算Q估计
		Q_table = self.network.forward(feed_batch) #Qtable
		#从Q_table中获得Q(s,a)也就是Q估计
		Q_predict = Q_table[np.arange(self.batch_size),action_batch]

		#计算Q现实
		with torch.no_grad():
			feed_next_batch = torch.from_numpy(self.transform_state(nextstate_batch)).to("cuda")
			Q_table_next = self.network.forward(feed_next_batch)
			Q_nextmax = torch.Tensor(self.batch_size).to("cuda")
			for i in range(self.batch_size):
				nextstate = nextstate_batch[i]
				legal_actions = Game2048.legal_moves(nextstate)
				Q_nextmax[i] = torch.max(Q_table_next[legal_actions]) if legal_actions else 0
			#Q_现实公式
			Q_label = torch.Tensor(reward_batch).cuda() + self.reward_decay * Q_nextmax

		#计算loss并更新
		loss = self.loss_fn(Q_predict, Q_label)
		print("\n{} th train starts with {} loss".format(self.train_steps,loss))
		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()

	def update_all_batch(self):
		all_loss = 0
		for i in range(self.memory_size,0,-self.batch_size):
			memory = list(self.memory)
			state_batch,action_batch,nextstate_batch,reward_batch,isover_batch = zip(*memory[i-self.batch_size:i])
			feed_batch = torch.from_numpy(self.transform_state(state_batch)).to("cuda")

			#计算Q估计
			Q_table = self.network.forward(feed_batch) #Qtable
			#从Q_table中获得Q(s,a)也就是Q估计
			Q_predict = Q_table[np.arange(self.batch_size),action_batch]

			#计算Q现实
			with torch.no_grad():
				feed_next_batch = torch.from_numpy(self.transform_state(nextstate_batch)).to("cuda")
				Q_table_next = self.network.forward(feed_next_batch)
				Q_nextmax = torch.Tensor(self.batch_size).to("cuda")
				for i in range(self.batch_size):
					nextstate = nextstate_batch[i]
					legal_actions = Game2048.legal_moves(nextstate)
					Q_nextmax[i] = torch.max(Q_table_next[legal_actions]) if legal_actions else 0
				#Q_现实公式
				Q_label = torch.Tensor(reward_batch).cuda() + self.reward_decay * Q_nextmax

			#计算loss并更新
			self.scheduler.step()
			loss = self.loss_fn(Q_predict, Q_label)
			all_loss += loss.item()
			self.optimizer.zero_grad()
			loss.backward()
			self.optimizer.step()
		print("\n{} th train ends with {} loss".format(self.train_steps,all_loss/(self.memory_size//self.batch_size)))

	def train(self,mode,modelpath=None):
		if modelpath:
			self.network.load_state_dict(torch.load(modelpath))
		while self.eposide < self.max_eposide:
			game = Game2048()
			state = game.matrix
			local_steps = 0
			# print(game.matrix)
			while True:
				# print("{}th before current_score {}".format(local_steps,game.score))
				# print(game.matrix,np.array_equal(game.matrix,state))
				actionList = self.chooseAction(state,e_greedy=True)
				action,reward = game.step(state,actionList)
				nextstate = game.matrix
				# print(state,game.score)
				self.memory.append((state,action,nextstate,reward,game.isover)) #保存状态
				# print("{}th after current_score {}".format(local_steps,game.score))
				# print(game.matrix,np.array_equal(game.matrix,nextstate))
				(local_steps,self.total_steps) = (local_steps+1,self.total_steps+1) #局部计数器和全局计数器加一
				if len(self.memory) >= self.memory_size:
					self.train_steps += 1
					print("{} th train's learning rate :{}".format(self.train_steps,self.scheduler.get_lr()))
					if mode == "random":
						self.update_randomsample()
					else:
						for i in range(1):
							self.update_all_batch()
						self.memory = deque(maxlen=self.memory_size)
					# print("the {}th training ends".format(self.train_steps))
					if self.train_steps % self.save_per_iter == 0:
						torch.save(self.network.state_dict(), './parameter{}.pkl'.format(self.train_steps // self.save_per_iter%20))
						print("successfully saved")
				#本轮游戏结束退出当前循环,否则开始下一步
				if game.isover: 
					break
				else:
					state = nextstate.copy()
			self.max_num = game.maxnum if game.maxnum>self.max_num else self.max_num
			self.max_score = game.score if game.score>self.max_score else self.max_score
			print("\nEposide{} finished with score:{} maxnumber:{} steps={} ,details:".format(self.eposide,game.score,game.maxnum,local_steps))
			print(game.matrix,"epsilon:{}".format(self.epsilon))
			print("up to now max score:{} max number:{}".format(self.max_score,self.max_num))
			self.record.append((self.eposide,game.score,game.maxnum,local_steps)) #每局游戏记录局号、一局走了几步、游戏分数
			#更新最优参数
			self.eposide += 1
		torch.save(self.network.state_dict(), './finally.pkl')

	def test(self,n_eposide):
		self.network.load_state_dict(torch.load('./parameter0.pkl'))
		for i in range(n_eposide):
			game = Game2048()
			state = game.matrix
			local_steps = 0
			while True:
				actionList = self.chooseAction(state,e_greedy=False)
				action,reward = game.step(state,actionList)
				# print(game.matrix)
				local_steps += 1
				nextstate = game.matrix
				if game.isover: 
					break
				else:
					state = nextstate.copy()
			print("{}th eposide : gamescore={} maxnumber={} steps = {}".format(self.eposide,game.score,game.maxnum,local_steps))
			self.record.append((self.eposide,game.score,game.maxnum,local_steps)) #每局游戏记录局号、一局走了几步、游戏分数
			print(game.matrix)
			self.eposide += 1
		print(sum(r[1] for r in self.record)/len(self.record))
		print(max(r[2] for r in self.record))

if __name__ == "__main__":
	agent = Agent(memory_size=5120,batch_size=512)
	agent.train(mode="batch")
	# agent = Agent()
	# agent.test(1000)

	