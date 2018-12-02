import numpy as np
import random
import sys
import tensorflow as tf
import tkinter

from abc import ABCMeta , abstractmethod

WEIGHT = 0.1
ETA = 0.1

class Layer(metaclass=ABCMeta):
	def __init__(this,input_num,output_num):
	   this.w = WEIGHT * np.random.randn(input_num,output_num)
	   this.b = WEIGHT * np.random.randn(output_num)
	   this.inputNum = input_num
	   this.outputNum = output_num
	   this.className = ""

	@abstractmethod
	def forward(this,x):
		pass

	@abstractmethod
	def backward(this,x):
		pass

	def update(this):
		this.w -= this.grad_w * ETA
		this.b -= this.grad_b * ETA

	def copy(this):
		layer = create(this.inputNum,this.outputNum,name=this.className)
		layer.b = this.b.copy()
		layer.w = this.w.copy()
		return layer

class LayerIdentity(Layer):
	def __init__(this,input_num,output_num):
		super().__init__(input_num,output_num)
		this.className = "identity"

	def forward(this,x):
		this.x = x
		this.y = np.dot(x,this.w) + this.b

	def backward(this,x):
		delta = this.y - x
		this.grad_w = np.dot(this.x.T,delta)
		this.grad_b = np.sum(delta,axis=0)
		
		this.grad_x = np.dot(delta,this.w.T)

class LayerSigmoid(Layer):
	def __init__(this,input_num,output_num):
		super().__init__(input_num,output_num)
		this.className = "sigmoid"

	def forward(this,x):
		this.x = x
		this.y = np.dot(x,this.w) + this.b
		this.y = 1/(1+(np.exp(-this.y)))

	def backward(this,x):  
		delta = x * (1 - this.y) * this.y
		this.grad_w = np.dot(this.x.T,delta)
		this.grad_b = np.sum(delta,axis=0)

		this.grad_x = np.dot(delta,this.w.T)

class LayerSoftmax(Layer):
	def __init__(this,input_num,output_num):
		super().__init__(input_num,output_num)
		this.className = "softmax"

	def forward(this,x):
		this.x = x
		this.y = np.dot(x,this.w) + this.b
		this.y = np.exp(this.y)/np.sum(np.exp(this.y),axis=1,keepdims=True)

	def backward(this,x):  
		delta = this.y - x

		this.grad_w = np.dot(this.x.T,delta)
		this.grad_b = np.sum(delta,axis=0)

		this.grad_x = np.dot(delta,this.w.T)

class LayerLeakyReLU(Layer):
	def __init__(this,input_num,output_num):
		super().__init__(input_num,output_num)
		this.className = "leakyReLU"

	def forward(this,x):
		this.x = x
		this.y = np.dot(x,this.w) + this.b
		this.y = np.where(this.y <= 0,0.01 * this.y,this.y)

	def backward(this,x):
		delta = np.where(this.y <= 0,0.01,1)

		this.grad_w = np.dot(this.x.T,delta)
		this.grad_b = np.sum(delta,axis=0)

		this.grad_x = np.dot(delta,this.w.T)

class LayerDropout(Layer):
	def __init__(this,p=0.5):
		this.p = p

	def forward(this,x):
		rand = np.random.rand(*x.shape)
		this.dropout = np.where(rand > this.p , 1 , 0)
		this.y = x * this.dropout

	def backward(this,grad_y):
		this.grad_x = grad_y * this.dropout

	def update(this):
		()

class Network():
	def __init__(this,network=[]):
		this.network = network
		this.y = 0

	def forward(this,x):
		this.network[0].forward(x)
		for i in range(1,len(this.network)):
			this.network[i].forward(this.network[i - 1].y)
		this.y = this.network[-1].y
		return this.y

	def backward(this,x):
		this.network[-1].backward(x)
		for i in range(len(this.network) - 2,-1,-1):
			this.network[i].backward(this.network[i + 1].grad_x)

	def update(this):
		for i in this.network:
			i.update()

	def add(this,layer):
		this.network.append(layer)

	def copy(this):
		newNetwork = []
		for i in range(len(this.network)):
			newNetwork.append(this.network[i].copy())
		return Network(newNetwork)

	def save(this,pas="."):
		for i in range(0,len(this.network)):
			np.savetxt(pas + '/w0' + str(i) + '.csv', this.network[i].w, delimiter=',')
			np.savetxt(pas + '/b0' + str(i) + '.csv', this.network[i].b, delimiter=',')

	def load(this,pas = "."):
		for i in range(0,len(this.network)):
			this.network[i].w = np.loadtxt(pas + '/w0' + str(i) + '.csv', delimiter=',')
			this.network[i].b = np.loadtxt(pas + '/b0' + str(i) + '.csv', delimiter=',')

def set(weight=None,eta=None):
	if weight is not None:
		WEIGHT = weight
	if eta is not None:
		ETA = eta

def create(inputNum,outputNum,name="identity"):
	if name == "identity":
		return LayerIdentity(inputNum,outputNum)
	elif name == "sigmoid":
		return LayerSigmoid(inputNum,outputNum)
	elif name == "softmax":
		return LayerSoftmax(inputNum,outputNum)
	elif name == "leakyReLU":
		return LayerLeakyReLU(inputNum,outputNum)
	else:
		return None
