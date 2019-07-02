import torch.nn as nn
import torch.nn.functional as F
import torch
import math

class Generator(nn.Module):
	def __init__(self,patch_size):
		super(Generator, self).__init__()
		
		
		self.output_size = patch_size
		self.s = math.ceil(self.output_size/16.0)*16
		self.s2, self.s4, self.s8, self.s16 = int(self.s/2), int(self.s/4), int(self.s/8), int(self.s/16)

		self.ngf = 16 # number of generator filters in first conv layer
		# encoder_1: [batch, 16, 16, 3] => [batch, 8, 8, ngf]
		
		self.lrelu = nn.LeakyReLU(negative_slope=0.2)
		self.relu = nn.ReLU()
		self.layernorm = nn.LayerNorm() ########1 required positional argument: 'normalized_shape'
		
		self.conv1 = nn.Conv2d(3,self.ngf,4,stride=[1,2],padding=0,bias=True)
		# image.get_shape()[-1]
		# stride=[1,2,2,1]
		self.conv2 = nn.Conv2d(self.ngf,self.ngf*2,4,stride=[1,2],padding=0,bias=True)
		self.conv3 = nn.Conv2d(self.ngf*2,self.ngf*4,4,stride=[1,2],padding=0,bias=True)
		self.conv4 = nn.Conv2d(self.ngf*4,self.ngf*8,4,stride=[1,2],padding=0,bias=True)
		
		self.deconv1 = nn.ConvTranspose2d(self.ngf*8, self.ngf*4, 4, stride=[1,2], padding=0, bias=True)
		self.deconv2 = nn.ConvTranspose2d(self.ngf*8, self.ngf*2, 4, stride=[1,2], padding=0, bias=True)
		self.deconv3 = nn.ConvTranspose2d(self.ngf*2, self.ngf, 4, stride=[1,2], padding=0, bias=True)
		self.deconv4 = nn.ConvTranspose2d(self.ngf*2, 3, 4, stride=[1,2], padding=0, bias=True)

	def forward(self, image):
		x = self.conv1(image)
		x = self.lrelu(x)
		x = self.conv2(x)
		x = self.layernorm(x)
		x = self.lrelu(x)
		x = self.conv3(x)
		x = self.layernorm(x)
		x = self.lrelu(x)
		x = self.conv4(x)
		x = self.layernorm(x)
		x = self.relu(x)
		x = self.deconv1(x)
		x = self.layernorm(x)
		input = torch.cat([self.deconv1, self.conv3], axis=3)
		input = self.relu(input)
		x = self.deconv2(input)
		x = self.layernorm(x)
		input = torch.cat([self.deconv2, self.conv2], axis=3)
		input = self.relu(input)
		x = self.deconv3(input)
		x = self.layernorm(x)
		input = torch.cat([deconv3, conv1], axis=3)
		input = self.relu(input)
		x = self.deconv4(input)
		
		return torch.nn.tanh(x)
		
if __name__ == '__main__':
	x = torch.randn(1,3,256,256)
	print(x.shape)
	G = Generator(1)
	out = G(x)
	print(out.shape)

