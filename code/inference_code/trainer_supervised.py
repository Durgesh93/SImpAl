import torch
import os
import logging
import glob
import numpy as np

import utils as utils
import metrics as metrics
import config as config
import torch.nn as nn

from tqdm import tqdm
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from net import MutiSourceNet as mmodel
from dataset import TemplateDataset



class TrainerG(object):

	def __init__(self):

		self.settings 									= config.settings
		self.network 									= mmodel().to(self.settings['device'])

		self.to_train 									= self.settings['to_train']
		self.val_batch_size 							= self.settings['val_batch_size']
		self.exp_name									= self.settings['exp_name']
		self.phase										= self.settings['mode']['train']
		self.target_dl_iter_val_list 					= {}
		self.grp_name_suff            					= 'val'


	'''
	Function to load model weights
	'''
	def load_weights_model(self):

		load_weights_path = os.path.join(self.settings['weights_path'],self.settings['exp_name'],'model_'+str(self.settings['iterno'])+'.pth')
		dict_to_load = torch.load(load_weights_path,map_location=self.settings['device'])
		model_state_dict = dict_to_load['model_state_dict']

		for nc,compts in self.network.model.items():
			for name,comp in compts.items():
				self.network.model[nc][name].load_state_dict(model_state_dict['_'.join([nc,name])])


	def get_all_val_target_dataloaders(self):
		for dom in self.settings['trgt_datasets']:
			self.initalize_target_val_dataloader(dom)


	def initalize_target_val_dataloader(self,dom):
		target_dataset_val = TemplateDataset('_'.join([dom,'test.npy']),aug=False)	
		self.target_dl_iter_val_list[dom] = iter(DataLoader(target_dataset_val, batch_size=self.val_batch_size, shuffle=False, num_workers=2,pin_memory=True))
	

	def set_mode(self,mode):
		self.phase = mode

		if self.phase == self.settings['mode']['train']:
			for nc,compts in self.to_train.items():
				for name,val in compts.items():
					if val:
						self.network.model[nc][name].train()
					else:
						self.network.model[nc][name].eval()

		elif self.phase == self.settings['mode']['val']:
			self.network.eval()


	def prepare_indices_domain_net(self):
		C 					= self.settings['C']
		C_dash 				= self.settings['C_dash']
		server_root_path    = self.settings['server_root_path']
		dataset_dir 		= self.settings['dataset_dir']
		trgt_datasets  		= self.settings['trgt_datasets']
		src_datasets        = self.settings['src_datasets']
		
		if not os.path.exists(os.path.join(server_root_path, dataset_dir,config.settings['exp_name'])):
			os.mkdir(os.path.join(server_root_path, dataset_dir,config.settings['exp_name']))

		if not os.path.exists(os.path.join(server_root_path,dataset_dir,config.settings['exp_name'],'index_list')):
			os.mkdir(os.path.join(server_root_path, dataset_dir,config.settings['exp_name'],'index_list'))

		shared_trgt_catgs =  set([j for x in src_datasets for j in C[x]])


		for dataset_name in trgt_datasets:
			imgs_path_val    = []
			cat_mapping = utils.get_catg_mapping(shared_trgt_catgs,C_dash[dataset_name])
			with open(os.path.join(server_root_path,dataset_dir,'index_main','_'.join([dataset_name,'test.txt'])),'r') as f:
				data = f.readlines()
				for line in data:
					img_path,clas_lbl = line.split(' ')
					_,catg = os.path.split(os.path.split(img_path)[0])
					if catg in shared_trgt_catgs or catg in C_dash[dataset_name]:
						cat_id 	= cat_mapping[catg]
						imgs_path_val.append([os.path.join(server_root_path,dataset_dir,img_path.strip()),cat_id])

			save_path = os.path.join(server_root_path, dataset_dir,config.settings['exp_name'],'index_list', '_'.join([dataset_name,'test.npy']))
			np.save(save_path,imgs_path_val)


	def prepare_indices_generic(self):

		C 					= self.settings['C']
		C_dash 				= self.settings['C_dash']

		server_root_path    = self.settings['server_root_path']
		dataset_dir 		= self.settings['dataset_dir']

		src_datasets  		= self.settings['src_datasets']
		trgt_datasets  		= self.settings['trgt_datasets']
		resolution 			= self.settings['resolution']


		if not os.path.exists(os.path.join(server_root_path,dataset_dir, config.settings['exp_name'])):
			os.mkdir(os.path.join(server_root_path, dataset_dir,config.settings['exp_name']))


		if not os.path.exists(os.path.join(server_root_path,dataset_dir, config.settings['exp_name'], config.settings['index_list'])):
			os.mkdir(os.path.join(server_root_path,dataset_dir, config.settings['exp_name'], config.settings['index_list']))


		shared_trgt_catgs =  set([j for x in src_datasets for j in C[x]])

		for dataset_name in trgt_datasets:
			imgs_paths = []
			cat_mapping = utils.get_catg_mapping(shared_trgt_catgs,C_dash[dataset_name])

			for catg in  os.listdir(os.path.join(server_root_path, dataset_dir, dataset_name)):
				if catg in shared_trgt_catgs or catg in C_dash[dataset_name]:
					cat_id 	= cat_mapping[catg]
					imgs_paths.extend([[x,cat_id] for x in glob.glob(os.path.join(server_root_path, dataset_dir, dataset_name,catg,'*'))])
			
			save_path = os.path.join(server_root_path, dataset_dir,config.settings['exp_name'], config.settings['index_list'], '_'.join([dataset_name,'test.npy']))
			np.save(save_path,imgs_paths)


	'''
	Target dataset validation
	'''
	def val_over_target_set(self):
		self.get_all_val_target_dataloaders()

		with torch.no_grad():
			for dom in self.settings['trgt_datasets']:
				
				all_labels_trgt 				 =  []
				all_preds_trgt  				 =  []				
				target_dl_iter_val_list = self.target_dl_iter_val_list[dom]
				

				for data in tqdm(target_dl_iter_val_list,desc=dom):

					indx,images,label,_  		= data

					x				 			= images.to(self.settings['device']).float()
					label			 			= label.to(self.settings['device']).long()

					G 							= self.network.model['global']['G'](x)
					F 							= self.network.model['global']['Fs'](G)
					M 							= self.network.model['global']['M'](F)


					cls_logits,_,mat 			= metrics.get_logits(key=self.settings['target_label_logit_key'],feats={'M':M})
					cls_confs,cls_preds    		= torch.max(cls_logits,dim=-1)

					all_labels_trgt.append(label)
					all_preds_trgt.append(cls_preds)

				all_preds_trgt = torch.cat(all_preds_trgt)
				all_labels_trgt = torch.cat(all_labels_trgt)

				acc = (all_preds_trgt == all_labels_trgt).float().sum()/all_preds_trgt.shape[0]
				print('target domain {} acc = {}'.format(dom,acc))


if __name__=='__main__':
	raise NotImplementedError('Please check train file')

