import torch.nn as nn
import torch
from customlayers import MatrixLayer as M
from customlayers import BackBoneLayer as G
from customlayers import IdentityLayer as I
from customlayers import ForwardLayer as F
import config as config


class MutiSourceNet(nn.Module):
	
	def __init__(self): 
		
		super(MutiSourceNet, self).__init__()
		
		self.model = {}
		to_train = config.settings['to_train']
		
		for nc,compts in to_train.items():
			self.model[nc]={}

			for name in compts:
				if name =='Fs':
					self.model[nc][name] = F(config.settings['bb_output'],config.settings['bb_output']//2,config.settings['F_dims'])
				elif name =='M':
					self.model[nc][name] = M(config.settings['F_dims'],len(config.settings['model_dict']['src_datasets']),config.settings['num_C'])
				elif name =='G':
					self.model[nc][name] = G(config.settings['bb'],config.settings['bb_output'])

		for nc,compts in self.model.items():
			for name,comp in compts.items():
				self.add_module('_'.join([nc,name]),comp)


	def forward(self, x ):
		raise NotImplementedError('Implemented a custom forward in train loop')


	def no_param(self):
		param = 0
		for p in list(self.parameters()):
			n=1
			for i in list(p.size()):
				n*= i
			param += n
		return param


if __name__=='__main__':
	raise NotImplementedError('Please check README.md for execution details')
