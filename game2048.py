'''
Author:qinyuheng
Mail:qinyuheng@hhu.edu.cn
Github:QYHcrossover
Date:2020/06/06
'''
import numpy as np
import random
import os

class Game2048:
	def __init__(self,matrix_size=4):
		self.matrix_size = matrix_size #方格尺寸
		self.matrix = np.zeros([self.matrix_size,self.matrix_size],dtype=np.int64) #方格矩阵
		self.score = 0 #游戏得分
		#游戏初始化，随机生成两个方块
		self.generate() 
		self.generate()

	def generate(self): #产生新方块
		x,y = np.where(self.matrix==0)
		index = np.random.randint(len(x))
		self.matrix[x[index]][y[index]] = 2 if random.random() < 0.9 else 4

	@staticmethod
	def move(state,action): #移动并合并
		# assert action in ["up","down","left","right"]
		def move_one(array): #单列移动、合并,并返回sorce
			origin = array.copy()
			score = 0
			index  = np.where(array)[0]
			(before,after) = (0,1)
			while after < len(index):
				if array[index[before]] == array[index[after]]:
					array[index[before]] *= 2
					score += array[index[before]]
					array[index[after]] = 0
				(before,after) = (after,after+1)
			extracted = array[array!=0]
			padded = np.zeros(4,dtype=np.int64)
			padded[:len(extracted)] = extracted
			return not (origin==padded).all(),padded,score

		movescore = 0
		ismove = False
		if action in ("w","s",0,1):
			for i in range(4):
				feed = state[:,i] if action in ("w",0) else state[:,i][::-1]
				isonemove,array,onescore = move_one(feed)
				ismove = ismove or isonemove
				state[:,i] = array if action in ("w",0) else array[::-1]
				movescore += onescore

		if action in ("a","d",2,3):
			for i in range(4):
				feed = state[i] if action in ("a",2) else state[i][::-1]
				isonemove,array,onescore = move_one(feed)
				ismove = ismove or isonemove
				state[i] = array if action in ("a",2) else array[::-1]
				movescore += onescore
		return ismove,movescore,state

	@property
	def maxnum(self):
		return np.max(self.matrix)
	
	@property
	def isover(self):
		for i in range(self.matrix_size):
			for j in range(self.matrix_size):
				if self.matrix[i][j] == 0: return False
				if (i == self.matrix_size - 1) and (j == self.matrix_size - 1):
					continue
				elif (i == self.matrix_size - 1):
					if (self.matrix[i][j] == self.matrix[i][j+1]):
						return False
				elif (j == self.matrix_size - 1):
					if (self.matrix[i][j] == self.matrix[i+1][j]):
						return False
				else:
					if (self.matrix[i][j] == self.matrix[i+1][j]) or (self.matrix[i][j] == self.matrix[i][j+1]):
						return False
		return True

	#命令行版本的运行
	def run(self):
		while True:
			os.system("cls")
			print(self.matrix)
			print("current score：{}".format(self.score))
			if self.isover:
				break
			action = input("").strip().lower()
			ismove,movescore,nextstate = self.move(self.matrix,action)
			self.matrix = nextstate
			self.score += movescore
			if ismove:
				self.generate()

	@staticmethod
	def legal_moves(state):
		lms = []
		for action in range(4):
			ismove = Game2048.move(state,action)[0]
			if ismove:
				lms.append(action)
		return lms
	
	#为dqn准备的API
	def step(self,state,actionList):
		index = np.argsort(-actionList)
		for action in index:
			ismove,movescore,nextstate = self.move(state,action)
			if ismove:
				trueAction = action
				break
		# print("movescore{}".format(movescore))
		self.matrix = nextstate
		self.score = self.score + movescore
		self.generate()
		#分别返回真实action和reward
		reward = 0.0 if movescore == 0 else np.log2(movescore)/15
		return trueAction,reward

if __name__ == "__main__":
	game = Game2048()
	game.run()
