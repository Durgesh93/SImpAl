import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import config as config
from tensorboardX import SummaryWriter
import numpy as np
import io
import logging
import shutil


def get_catg_mapping():
	src_C = config.settings['C'][config.settings['src_datasets'][0]]
	src_C_dash = [c  for dom in config.settings['src_datasets'] for c in config.settings['C_dash'][dom]]
	all_C = []
	all_C.extend(src_C)
	all_C.extend(src_C_dash)
	catg_mapping = {j:i for i,j in enumerate(all_C)}
	return catg_mapping


def get_domain_mapping(src_domains,trgt_domains):
	
	src_dm = [k for k in src_domains]
	trgt_dm = [k for k in trgt_domains]

	src_dm.sort()
	trgt_dm.sort()


	dom_mapping = {j:i for i,j in enumerate(src_dm)}

	for i,j in enumerate(trgt_dm):
		dom_mapping[j]=i+len(src_dm)
	
	return dom_mapping



class Summ(object):

	def __init__(self,grp,key,func,writer):
		super(Summ, self).__init__()
		self.curr_val 			= 0
		self.tot_val  			= 0
		self.tot_count  		= 0
		self.curr_iterno 		= 0
		self.curr_count 		= 0

		self.grp 				= grp
		self.key 	  			= key
		self.func  				= func
		self.writer 			= writer



	def update(self,curr_val,count,iterno):
		self.curr_val = curr_val

		if torch.is_tensor(curr_val):
			if curr_val.dim() == 0:
				self.curr_val = curr_val.item()
			else:
				self.curr_val = curr_val.cpu().numpy()

		self.tot_count			+=count
		self.tot_val  			+=self.curr_val
		self.curr_iterno         = iterno
		self.curr_count 		 = count


	def get_val(self):
		return self.func(self.curr_val,self.tot_val,self.curr_count,self.tot_count)

	def tb(self):
		self.writer.add_scalar(self.grp+'/'+self.key, self.get_val(), int(self.curr_iterno))



class PatienceSumm(object):

	def __init__(self, grp,key, mode, max_pat, writer):
		super(PatienceSumm, self).__init__()
		self.curr_val     = 0
		self.curr_iterno  = 0
		self.max_pat = max_pat
		self.curr_pat = 0
		self.mode = mode
		self.writer = writer
		self.key = key
		self.grp = grp

		if mode =='geq':
			self.best_val 			= -100000
		elif mode =='leq':
			self.best_val 			=  100000


	def update(self,curr_val,iterno):

		self.curr_val = curr_val

		if torch.is_tensor(curr_val):
			if curr_val.dim() == 0:
				self.curr_val = curr_val.item()
			else:
				self.curr_val = curr_val.cpu().numpy()

		self.curr_iterno = iterno

		if self.mode =='geq':
			val = max(self.curr_val,self.best_val)
		elif self.mode =='leq':
			val = min(self.curr_val,self.best_val)

		if val == self.curr_val:
			self.best_val = val
			self.curr_pat = 0
		else:
			self.curr_pat+=1

	def tb(self):
		self.writer.add_scalar(self.grp+'/'+self.key, self.curr_pat, int(self.curr_iterno))

