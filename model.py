import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np


class mlp_vae(nn.Module):

	
	def find_num_neurons(self,rho=1,is_enc=True):
		### Based on arithmetic sequence formula

		### d = (an-a1)/((self.num_layers_in/out)**rho)

		enc_dec_nums = []
		pair_list = []
		
		if (is_enc):
			a1 = self.input_size
			an = self.middle_size
			d = (an-a1)/((self.num_layers_in)**rho)
			num_in_list =  [int(a1) + i*d for i in range(0,self.num_layers_in+1)]
			pair_list = [num_in_list[i:i+2] for i in range(0,self.num_layers_in) ]

		else:
			a1 = self.middle_size
			an = self.input_size
			d = (an-a1)/((self.num_layers_out)**rho)
			num_in_list =  [int(a1) + i*d for i in range(0,self.num_layers_out+1)]
			pair_list = [num_in_list[i:i+2] for i in range(0,self.num_layers_out) ]
			
		return pair_list
	def __init__(self,input_size , middle_size , num_layers_in = 3, sym = True, num_layers_out = 3):
		super(mlp_vae,self).__init__()

		self.input_size = input_size
		self.middle_size = middle_size
		self.num_layers_in = num_layers_in
		self.sym = sym
		assert self.num_layers_in >=1 , 'num_layers must be atleast 1'
		self.encoder = nn.Sequential()
		self.decoder = nn.Sequential()

		self.name_layer = []
		### tyring out network config where inbetween layers have num neurons around the mean of the middle and input layers
		if(sym):
			self.num_layers_out = self.num_layers_in
		else:
			self.num_layers_out = num_layers_out

		self.name_layer+=['IH'+str(i) for i in range(0,num_layers_in)] 
		
		self.name_layer+=['OH'+str(i) for i in range(self.num_layers_out-1,-1,-1)] 
		
		self.enc_neurons = self.find_num_neurons(is_enc = True)
		
		self.dec_neurons = self.find_num_neurons(is_enc = False)
		
		for idx,num in enumerate(self.enc_neurons+self.dec_neurons):
			#print(idx)
			if('IH' in self.name_layer[idx]):
				self.encoder.add_module(self.name_layer[idx],nn.Linear(int(num[0]),int(num[1])))
			else:
				self.decoder.add_module(self.name_layer[idx],nn.Linear(int(num[0]),int(num[1])))

	def forward(self,x):

		low_dim_rep = self.encoder(x)


		return low_dim_rep,self.decoder(low_dim_rep)


