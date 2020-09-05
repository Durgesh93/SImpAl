import torch
import utils
import torch.nn as nn
import config as config
from customlayers import MatrixLayer as M
from customlayers import BackBoneLayer as G
from customlayers import ForwardLayer as F

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
					num_C = len(utils.get_catg_mapping())
					self.model[nc][name] = M(config.settings['F_dims'],len(config.settings['src_datasets']),num_C)
				elif name =='G':
					self.model[nc][name] = G(config.settings['bb'],config.settings['bb_output'])

		for nc,compts in self.model.items():
			for name,comp in compts.items():
				self.add_module('_'.join([nc,name]),comp)


	def forward(self, x ):
		raise NotImplementedError('Implemented a custom forward in train loop')


if __name__=='__main__':
	raise NotImplementedError('Please check README.md for execution details')
