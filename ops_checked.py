import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

WEIGHTS_INIT_STDEV = .1

def lrelu(x, leak=0.2):#checked
	return torch.max(x, leak * x)
'''
input = torch.zeros([2,2]) - torch.ones([2,2])
print(lrelu(input))
'''

# layer normalization
def layer_norm(shape):#checked
	layer = nn.LayerNorm(normalized_shape=shape)
	return layer
'''
input = torch.randn(20,5,10,10)
print(layer_norm(input.size()[1:]))
'''

def linear(input_, output_size, stddev=0.02, bias_start=0.0, with_w=False):#checked
	shape = input_.size()
	print(shape)
	matrix = nn.init.normal_(torch.empty([shape[1], output_size], dtype=torch.float32), stddev)
	bias = nn.init.constant_(torch.empty([output_size]), bias_start)
	if with_w:
		return torch.matmul(input_, matrix) + bias, matrix, bias
	else:
		return torch.matmul(input_, matrix) + bias
'''
input = torch.randn(20,5,10,10)
batch_size = 10
print(linear(input.reshape(batch_size,-1),10))
'''