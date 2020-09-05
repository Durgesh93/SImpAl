import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import os
from torch.utils.data import DataLoader
from dataset import TemplateDataset
from tensorboardX import SummaryWriter
from net import MutiSourceNet as mmodel
import utils as utils
import metrics as metrics
import config as config
from net import MutiSourceNet as mmodel
from tqdm import tqdm
from torch.utils.data import ConcatDataset
import logging
from torch.optim.lr_scheduler import LambdaLR
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import SGDClassifier
from sklearn.mixture import GaussianMixture
import matplotlib
import matplotlib.pyplot as plt
import cv2
from utils import get_catg_mapping


class TrainerG(object):

	def __init__(self):


		self.settings 									= config.settings
		self.network 									= mmodel().to(self.settings['device'])
		self.to_train 									= self.settings['to_train']
		self.val_batch_size 							= self.settings['val_batch_size']		
		self.phase										= self.settings['mode']['val']
		self.data_dump 									= {}
		self.target_dataset_dl 							= {}
		self.query_dataset_dl						    = {}

		

	def prepare_indices_domain_net(self):
		C 					= self.settings['C']
		server_root_path    = self.settings['server_root_path']
		dataset_dir 		= self.settings['dataset_dir']
		target_dataset      = self.settings['target_dataset']

		
		if not os.path.exists(os.path.join(server_root_path, dataset_dir,self.settings['model_dict']['exp_name'])):
			os.mkdir(os.path.join(server_root_path, dataset_dir,self.settings['model_dict']['exp_name']))

		if not os.path.exists(os.path.join(server_root_path,dataset_dir,self.settings['model_dict']['exp_name'],'index_list')):
			os.mkdir(os.path.join(server_root_path, dataset_dir,self.settings['model_dict']['exp_name'],'index_list'))

		cat_mapping = get_catg_mapping(C)
				
		imgs_path_val = []

		with open(os.path.join(server_root_path,dataset_dir,'index_main','_'.join([target_dataset,'test.txt'])),'r') as f:
			data = f.readlines()
			for line in data:
				img_path,clas_lbl = line.split(' ')
				_,catg = os.path.split(os.path.split(img_path)[0])
				if catg in C:
					cat_id 	= cat_mapping[catg]
					imgs_path_val.append([os.path.join(server_root_path,dataset_dir,img_path.strip()),cat_id])

		imgs_path = imgs_path_val
		save_path = os.path.join(server_root_path, dataset_dir,config.settings['model_dict']['exp_name'],'index_list', '_'.join([target_dataset,'full.npy']))
		np.save(save_path,imgs_path)
	


	def load_weights_model(self):

		load_weights_path = os.path.join(self.settings['server_root_path'],'weights',self.settings['model_dict']['exp_name'],'model_'+str(self.settings['model_dict']['iterno'])+'.pth')
		dict_to_load = torch.load(load_weights_path,map_location=self.settings['device'])
		model_state_dict = dict_to_load['model_state_dict']

		for nc,compts in self.network.model.items():
			for name,comp in compts.items():
				self.network.model[nc][name].load_state_dict(model_state_dict['_'.join([nc,name])])
	

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

	def concat_tile(self,im_list_2d):
		return cv2.vconcat([cv2.hconcat(im_list_h) for im_list_h in im_list_2d])



	def get_tile(self,img_list):
		tile_shape = (self.settings['num_imgs'],self.settings['num_imgs'])
		if len(img_list) !=tile_shape[0]*tile_shape[1]:
			return
		img_list_arr = []
		for i in range(tile_shape[0]):
			img_list_arr.append([])
			for j in range(tile_shape[1]):
				img_list_arr[i].append(255.0*img_list[i*tile_shape[1]+j].permute(1,2,0).cpu().numpy())
		
		img_tile = self.concat_tile(img_list_arr)
		return img_tile




	def initialize_target_dataset(self,dom):
		target_dataset = TemplateDataset('_'.join([dom,'full.npy']),aug=False)	
		target_dl = iter(DataLoader(target_dataset, batch_size=self.val_batch_size, shuffle=False, num_workers=5,pin_memory=True))
		self.target_dataset_dl[dom]= target_dl


	def get_target_logits(self):

		with torch.no_grad():

			all_target_logits = []
			all_target_imgs   = []

			for data in tqdm(self.target_dataset_dl[self.settings['target_dataset']],desc=self.settings['target_dataset']):

				indx,images,label  			= data
				
				x				 			= images.to(self.settings['device']).float()
				label			 			= label.to(self.settings['device']).long()

				G 							= self.network.model['global']['G'](x)
				F							= self.network.model['global']['Fs'](G)
				M 							= self.network.model['global']['M'](F)
				cls_logits,_,mat 			= metrics.get_logits(key=self.settings['target_label_logit_key'],feats={'M':M})
				cls_confs,cls_preds    		= torch.max(cls_logits,dim=-1)
				all_target_logits.append(cls_logits)
				all_target_imgs.append(images)

			self.all_target_logits = torch.cat(all_target_logits,dim=0)
			self.all_target_imgs   = torch.cat(all_target_imgs,dim=0)


	def get_nearest_target_img(self,img,num_imgs):
		
		with torch.no_grad():
			
			image 						= torch.stack([img],dim=0)
			x				 			= image.to(self.settings['device']).float()
			G 							= self.network.model['global']['G'](x)
			F							= self.network.model['global']['Fs'](G)
			M 							= self.network.model['global']['M'](F)
			cls_logits,_,mat 			= metrics.get_logits(key=self.settings['target_label_logit_key'],feats={'M':M})
			
			target_pred_img   = []
			query_logit       = cls_logits
			
			dist_logit = torch.norm(self.all_target_logits - query_logit,dim=-1)
			idx = torch.argsort(dist_logit,dim=-1)
			target_imgs = self.all_target_imgs[idx[:num_imgs]]

			for target_img in target_imgs:
				target_pred_img.append(target_img)
			
		return np.uint8(self.get_tile(target_pred_img))



			
			

			


if __name__=='__main__':
	raise NotImplementedError('Please check train file')
