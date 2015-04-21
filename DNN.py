# coding=UTF-8

import numpy as np
#import theano as T
import random
import sys
import json
import time


class Layer:
	def __init__(self, n_in, n_out, W=None, b=None):
		self.n_in = n_in
		self.n_out = n_out

		if W == None and b == None:
			'''
			randomly initialize W and b
			'''
			self.b = np.zeros(n_out)#, dtype=T.config.floatX)
			self.W = np.asarray(
				np.random.uniform(
					low=-4*np.sqrt(6. / (n_in + n_out)),
					high=4*np.sqrt(6. / (n_in + n_out)),
					size=(n_out, n_in)
				)#,
				#dtype=T.config.floatX
			)
		else:
			self.b = np.array(b)
			self.W = np.array(W)

		self.z = np.zeros(n_out)#, dtype=T.config.floatX)
		self.a = np.full(n_out, 0.5)#, dtype=T.config.floatX)


	def forward(self, x):
		#if x>350, function would fail
		def tanh(x):
			return (np.exp(2.*x)-1.) / (np.exp(2.*x)+1.)
		'''
		given input x, compute z and a
		'''
		self.x = x
		self.z = np.dot(self.W, x)+self.b
		self.a = tanh(self.z)
		#self.a = 1. / (1. + np.exp(-self.z))
		return self.a


class DNN:
	def __init__(self, n_in, struct, n_out, batch_size=10, learning_rate=0.008, n_epoch=None, model=None):
		if model == None:
			self.n_in = n_in
			self.n_out = n_out
			self.struct = map( lambda n: int(n), struct.split("X") ) #self.struct = [? ? ?]
			self.batch_size = batch_size
			self.learning_rate = learning_rate
			self.n_epoch = n_epoch

			'''
			construct hidden layer and output layer
			'''
			self.layers = []
			#input layer
			self.layers.append(Layer(n_in, self.struct[0]))
			#hidden layer
			for i in range(1, len(self.struct)):
				layer = Layer(self.struct[i-1], self.struct[i])
				self.layers.append(layer)
			#output layer
			self.layers.append(Layer(self.struct[-1], n_out))
			#number of layers
			self.n_layer = len(self.layers)
		else:
			'''
			load a existing model
			'''
			self.n_in = model['n_in']
			self.n_out = model['n_out']
			self.batch_size = batch_size
			self.learning_rate = learning_rate
			self.n_epoch = n_epoch
			self.n_layer = model['n_layer']
			self.layers = []
			layers = model['layers']
			for i in range(len(layers)):
				W = layers[i]['W']
				b = layers[i]['b']
				self.layers.append(Layer(len(W[0]), len(W), W, b))

	def forward(self, x):
		'''
		given input x, output a
		'''
		now = np.array(x)#, dtype=T.config.floatX)
		for i in range(self.n_layer):
			now = self.layers[i].forward(now)
		return now


	def backpropogate(self, x, y, true_y):

		def d_tanh(x):
			return 4. / (np.exp(2.*x)+np.exp(-2.*x)+2.)

		'''
		def d_sig(z):
			#differentiate sigmoid(z)
			exp_z = np.exp(z)
			return exp_z / ((1 + exp_z)**2)
		'''
		'''
		backpropogation
		'''
		for i in range(self.n_layer-1, -1, -1):
			layer = self.layers[i]
			if i == self.n_layer-1: #最後一層
				dE_dy = y - true_y
				layer.delta = d_tanh(layer.z) *  dE_dy
				layer.grad_b = layer.delta

				layer.grad_W = layer.delta[:, None] * layer.x
			else:
				layer.delta = d_tanh(layer.z) * np.dot(self.layers[i+1].W.T, self.layers[i+1].delta)
				layer.grad_W = layer.delta[:, None] * layer.x
				layer.grad_b = layer.delta


	def train(self, data):

		size = len(data)

		#all_y = train_labels  all_x = train_data   (string)
		all_y, all_x = zip(*data)

		#open 48Num to 39num.map & construct 48=>39 look_up_table
		file_tem = open("./transfer table/48Num to 39num.map", "r")
		look_up_table = []
		for line in file_tem:
			#aa\t1\t4
			tem = line.rstrip("\n").split("\t")
			tem.pop(0)
			look_up_table.append( [int(i) for i in tem] )

		#construct label vectors
		new_y = [np.zeros(self.n_out) for y in all_y]
		for i in range(size):
			for j in look_up_table[all_y[i]-1]:
				new_y[i][j-1] = 1.
		all_y = new_y
		#data = [ (train_label, train_data) ... ]
		data = zip(all_y, all_x)

		if self.n_epoch == None:
			n_epoch = 18
		else:
			n_epoch = self.n_epoch

		#previous val acc
		prev_val_acc = 0.

		#start training for n_epoch times
		for epoch in range(n_epoch):

			#record the start time
			t_start = time.time()

			random.shuffle(data)
			all_y, all_x = zip(*data)
			#for validation
			#val_y, val_x = all_y[:size/10], all_x[:size/10]

			'''
			TODO: mini-batch
			'''

			'''
			SGD process
			'''
			for i in range(size):
				x = all_x[i]
				true_y = all_y[i]
				y = self.forward(x)
				#backprop, store gradients in each layer
				self.backpropogate(x, y, true_y)
				'''
				update parameters
				'''
				for layer in self.layers:
					layer.W = layer.W - self.learning_rate * layer.grad_W
					layer.b = layer.b - self.learning_rate * layer.grad_b

			#record the finish time
			t_stop = time.time()

			'''
			output status of this epoch
			'''
			val_acc = self.evaluate(epoch, t_stop-t_start, all_x, all_y)
			#val_acc = self.evaluate(epoch, t_stop-t_start, all_x, all_y, val_x, val_y)

			#error function下降太少就直接early stopping
			'''
			if val_acc - prev_val_acc < 0.0001:
				self.write_model(name = "model_" + str(int(time.time())))
				break
			else:
				prev_val_acc = val_acc
				self.write_model(name = "model_" + str(int(time.time())))
			'''
		#model
		self.write_model(name = "model_" + str(int(time.time())))

	def evaluate(self, epoch, time, train_x, train_y):
	#def evaluate(self, epoch, time, train_x, train_y, val_x, val_y):

		#calculate Ein
		acc = 0.
		for i in range(len(train_x)):
			x = train_x[i]
			true_y = train_y[i]
			y = self.forward(x)
			if true_y[np.argmax(y)] == 1.:
				acc += 1
		print "#epoch{0} - {1:.1f}sec\n  in-acc: {2}".format(epoch, time, acc/len(train_x))
		'''
		acc = 0.
		for i in range(len(val_x)):
			x = val_x[i]
			true_y = val_y[i]
			y = self.forward(x)
			if true_y[np.argmax(y)] == 1.:
				acc += 1
		print "  val-acc: : {0}".format(acc/len(val_x))
		'''
		return acc/len(train_x)

	def write_model(self, name="dnn.model"):
		f = open(name, "w")
		model = {
			"n_in": self.n_in,
			"n_out": self.n_out,
			"n_layer": self.n_layer,
			"layers": []
		}

		for i in range(self.n_layer):
			layer = self.layers[i]
			conf = {
				"n_in": layer.n_in,
				"n_out": layer.n_out,
				"W": layer.W.tolist(),
				"b": layer.b.tolist()
			}
			model["layers"].append(conf)

		json.dump(model, f, indent=1)
		f.close()

	def predict(self, data):
		'''
		predict labels of testing data
		'''
		ans = []
		for d in data:
			y = self.forward(d[1])
			ans.append((d[0], str(np.argmax(y)+1)))
		return ans

def learn(training_data, input_model_name=None):

	#open train data
	train_f = open("./data/"+training_data, "r")

	#data => list of tuples => [ (label, data) ]
	data = []
	for line in train_f:
		arr = line.split(" ")
		label = int(arr.pop(0))
		arr = [float(e) for e in arr]
		data.append((label, arr))
	train_f.close()

	#initial model setting
	if input_model_name == None:
		dnn = DNN(n_in=69, struct="1024X1024X1024X1024", n_out=48, batch_size=10, n_epoch=15, learning_rate=0.008)
	else:
		mf = open("./model/"+input_model_name, "r")
		model = json.load(mf)
		dnn = DNN(n_in=69, n_out=48, model=model, batch_size=16, n_epoch=15, learning_rate=0.008)
		mf.close()

	#train dnn
	dnn.train(data)
	#write model to "model_%time"
	#dnn.write_model(name = "model_" + str(int(time.time())))   #好像會重複輸出model

def predict(test_name, model_name):

	#load model
	mf = open("./model/"+model_name, "r")
	model = json.load(mf)
	dnn = DNN(model=model)

	#load test data
	tf = open("./data/"+test_name, "r")

	#data = [ (pid, data) ]
	data = []
	for line in tf:
		arr = line.split(" ")
		pid = arr.pop(0)
		arr = [float(e) for e in arr]
		data.append((pid, arr))

	#close model & testdata
	mf.close()
	tf.close()

	print len(data)

	return dnn.predict(data)
