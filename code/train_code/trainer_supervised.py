import torch
import os
import logging
import numpy as np

import utils as utils
import metrics as metrics
import config as config
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from tabulate import tabulate
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from net import MutiSourceNet as mmodel
from dataset import TemplateDataset,PseudoTargetDataset



logger = logging.getLogger()

class TrainerG(object):

	def __init__(self):

		self.settings 									= config.settings
		self.network 									= mmodel().to(self.settings['device'])

		self.to_train 									= self.settings['to_train']

		#summary weiters
		self.test_writer 								= SummaryWriter(os.path.join(self.settings['server_root_path'], 'summaries', self.settings['exp_name'] , 'logdir_test'))
		self.train_writer								= SummaryWriter(os.path.join(self.settings['server_root_path'], 'summaries', self.settings['exp_name'] , 'logdir_train'))


		#batch size
		self.batch_size 								= self.settings['batch_size']
		self.val_batch_size 							= self.settings['val_batch_size_factor']*self.settings['batch_size']
		

		self.current_iteration  						= self.settings['start_iter']
		self.exp_name									= self.settings['exp_name']
		self.phase										= self.settings['mode']['train']
		self.curr_topK 									= self.settings['topK']

		#data loader dictionaries
		self.source_dl_iter_train_list 					= {}
		self.target_dl_iter_train_list                  = {}
		
		self.source_dl_iter_val_list 					= {}
		self.target_dl_iter_val_list 					= {}

		#data structure to hold pseudo target labels
		self.pseudo_target_dl_iter_train_list           = {}
		
		self.pseudo_indx_list 							= {}
		self.pseudo_cls_label_list 						= {}
		self.pseudo_logit_diff_list                     = {}
		self.pseudo_conf_list 		                    = {}

		self.expt_data 									= {}

		self.itt_delete 								= []
		self.max_test_acc								= -10000

		if self.settings['continue_training']:
			self.psed_orride								= True 
		else:
			self.psed_orride								= False 

		self.grp_name_suff             = 'val'
		self.get_all_train_src_dataloaders()
		self.init_optimizers()

		all_losses 		= self.optimizer_dict.keys()
		self.active_losses      = [current_loss for current_loss in all_losses if  self.settings['use_loss'][current_loss]]


	'''
	Utility function to implement logic while saving weights
	'''

	def check_and_save_weights(self,curr_cls_acc,dom):
		self.max_test_acc = max(curr_cls_acc,self.max_test_acc)
		if self.max_test_acc == curr_cls_acc:
			self.save_weights()
			self.itt_delete.append(self.current_iteration)
			if(len(self.itt_delete)>self.settings['checkpoints_count']):
				for k in self.itt_delete[:-self.settings['checkpoints_count']]:
					os.remove(os.path.join(self.settings['weights_path'],self.exp_name, 'model_' + str(k) + '.pth'))
					os.remove(os.path.join(self.settings['weights_path'],self.exp_name, 'opt_' + str(k) + '.pth'))
				self.itt_delete = self.itt_delete[-self.settings['checkpoints_count']:]

		if self.current_iteration == self.settings['enough_iter']:
			self.save_weights()


	'''
	Function to load model weights
	'''
	def load_weights_model(self):

		load_weights_path = os.path.join(self.settings['weights_path'],self.settings['model_dict']['exp_name'],'model_' + str(self.settings['model_dict']['iter']) + '.pth')
		dict_to_load = torch.load(load_weights_path,map_location=self.settings['device'])
		model_state_dict = dict_to_load['model_state_dict']

		for nc,compts in self.network.model.items():
			for name,comp in compts.items():
				self.network.model[nc][name].load_state_dict(model_state_dict['_'.join([nc,name])])

		expt_dict = torch.load(os.path.join(self.settings['weights_path'],self.exp_name,'expt_data.pth'),map_location=self.settings['device'])
		self.max_test_acc = expt_dict['max_test_acc']


	'''
	Function to load optimizer
	'''

	def load_optimizers(self):

		load_weights_path = os.path.join(self.settings['weights_path'],self.settings['opt_dict']['exp_name'],'opt_' + str(self.settings['opt_dict']['iter']) + '.pth')
		dict_to_load = torch.load(load_weights_path,map_location=self.settings['device'])
		optimizer_state_dict = dict_to_load['optimizer_state_dict']

		for name,optimizer in self.optimizer_dict.items():
			if self.settings['use_loss'][name]:
				optimizer.load_state_dict(optimizer_state_dict[name])
	

	'''
	Function to save model and optimizer state
	'''

	def save_weights(self):
		logger.msg('saving best weight at iteration number ={}'.format(self.current_iteration))
		weights_path = self.settings['weights_path']

		model_state_dict={}
		for nc,compts in self.network.model.items():
			for name,comp in compts.items():
				model_state_dict['_'.join([nc,name])]=comp.cpu().state_dict()

		optimizer_state_dict ={}
		for name,optimizer in self.optimizer_dict.items():
			optimizer_state_dict[name]=optimizer.state_dict()


		save_dict 	 = {
						 'model_state_dict':model_state_dict,
					   }


		if not os.path.exists(os.path.join(self.settings['weights_path'],self.exp_name)):
			os.mkdir(os.path.join(self.settings['weights_path'],self.exp_name))

		if self.current_iteration == self.settings['enough_iter']:
			torch.save(save_dict, os.path.join(self.settings['weights_path'],self.exp_name, 'model_enough_iter' + str(self.current_iteration) + '.pth'))

		torch.save(save_dict, os.path.join(self.settings['weights_path'],self.exp_name, 'model_' + str(self.current_iteration) + '.pth'))
		self.network.to(self.settings['device'])


		save_dict 	 = {
						 'optimizer_state_dict':optimizer_state_dict,
					   }


		if self.current_iteration == self.settings['enough_iter']:
			torch.save(save_dict, os.path.join(self.settings['weights_path'],self.exp_name, 'opt_enough_iter' + str(self.current_iteration) + '.pth'))

		torch.save(save_dict, os.path.join(self.settings['weights_path'],self.exp_name, 'opt_' + str(self.current_iteration) + '.pth'))
		self.network.to(self.settings['device'])


	'''
	Calculate pseudo target labels
	'''

	def initialize_pseudo_target_indices(self):
		print('Initializing pseudo target dataset in val mode')

		curr_set_mode = self.phase

		self.set_mode(config.settings['mode']['val'])

		calc_lbl = False

		if self.current_iteration < max(self.settings['val_after'],self.settings['enough_iter']):
			calc_lbl = False

		elif self.psed_orride:	
			calc_lbl = True
			self.psed_orride = False
		else:
			calc_lbl = True

		if not calc_lbl:
			return

			
		logger.msg('calculating pseudo labels after iteration ={}'.format(self.current_iteration))
		with torch.no_grad():
			for dom in self.settings['trgt_datasets']:
				
				target_dataset_train = TemplateDataset('_'.join([dom,'train.npy']),aug=False)

				target_dl_iter_train_list = iter(DataLoader(target_dataset_train, batch_size=self.val_batch_size, shuffle=False, num_workers=5,pin_memory=True))

				all_pseudo_labels 		= []
				all_indices     		= []
				all_M           		= []


				for data in tqdm(target_dl_iter_train_list):
					indx,images,_,_  			= data
					x				 			= images.to(self.settings['device']).float()

					G 							= self.network.model['global']['G'](x)
					F 							= self.network.model['global']['Fs'](G)
					M 							= self.network.model['global']['M'](F)

					cls_logits,_,_ 			= metrics.get_logits(key=self.settings['target_label_logit_key'],feats={'M':M})
					_,cls_preds    		    = torch.max(cls_logits,dim=-1)

					all_pseudo_labels.append(cls_preds)
					all_M.append(M)
					all_indices.append(indx)


				# N_points xDXC         M Matrix
				all_M 			  = torch.cat(all_M,dim=0)
				all_pseudo_labels = torch.cat(all_pseudo_labels,dim=0)
				all_indices       = torch.cat(all_indices,dim=0)
				

				K = torch.argmax(all_M,dim=-1)

				#checking for indices  where both classifiers agree
				idx1 = ((K == K[:,0].unsqueeze(dim=-1)).float().prod(dim=-1) == 1)

				#get sorted cls logits in decreasing order for each target sample and take effective logit distance ( most conf logit - second most conf logit)
				cls_logits,_                    = torch.sort(all_M,dim=-1,descending=True) # B x D x C
				all_logit_diffs 			    = (cls_logits[:,:,0] - cls_logits[:,:,1]).mean(dim=1)
			
				#getting logit difference where both classifier agree on prediction labels
				all_logit_diffs                 = all_logit_diffs[idx1]
				K 								= K[idx1]

				all_indices 					= all_indices[idx1]
				all_pseudo_labels               = all_pseudo_labels[idx1]
				
				idx 							= torch.argsort(all_logit_diffs,descending=True)
				all_indices 					= all_indices[idx]		
				all_pseudo_labels               = all_pseudo_labels[idx]
				all_logit_diffs					= all_logit_diffs[idx]

				
				min_logit_diff                  = all_logit_diffs.min()
				max_logit_diff                  = all_logit_diffs.max()

				all_logit_diffs_norm             = all_logit_diffs/max_logit_diff


				self.pseudo_cls_label_list[dom]  = all_pseudo_labels
				self.pseudo_indx_list[dom] 		 = all_indices
				self.pseudo_logit_diff_list[dom] = all_logit_diffs
				self.pseudo_conf_list[dom]       = all_logit_diffs_norm

		self.set_mode(curr_set_mode)
		print('setting model in mode{}'.format(curr_set_mode))
	

	'''
	Initializing optimizers
	'''
	def init_optimizers(self):
		self.optimizer_dict  = {}
		to_train = self.settings['to_train']
		for loss_name,loss_details in self.settings['optimizer_dict'].items():
			if self.settings['use_loss'][loss_name]:
				opt_param_list = []
				for dom,cmpts in loss_details.items():
						for comp in cmpts:
							if to_train[dom][comp]:
								if comp == 'G':
									opt_param_list.append({'params':self.network.model[dom][comp].parameters(), 'lr':self.settings['lr'][loss_name] / 10.0, 'weight_decay':5e-4})			
								else:
									opt_param_list.append({'params':self.network.model[dom][comp].parameters(), 'lr':self.settings['lr'][loss_name], 'weight_decay':5e-4})			
				self.optimizer_dict[loss_name] = optim.Adam(params = opt_param_list)


	'''
	Utility Functions to initalize source and target dataloaders
	'''
	def  get_all_train_src_dataloaders(self):
		for dom in self.settings['src_datasets']:
			self.initalize_src_train_dataloader(dom)

	def initalize_src_train_dataloader(self,dom):
		source_dataset_train = TemplateDataset('_'.join([dom,'train.npy']),aug=True)
		self.source_dl_iter_train_list[dom] = iter(DataLoader(source_dataset_train, batch_size=self.batch_size, shuffle=True, num_workers=5,drop_last=True,pin_memory=True))


	def get_all_val_target_dataloaders(self):
		for dom in self.settings['trgt_datasets']:
			self.initalize_target_val_dataloader(dom)


	def initalize_target_val_dataloader(self,dom):
		target_dataset_val = TemplateDataset('_'.join([dom,'test.npy']),aug=False)	
		self.target_dl_iter_val_list[dom] = iter(DataLoader(target_dataset_val, batch_size=self.val_batch_size, shuffle=False, num_workers=5,pin_memory=True))
	

	'''
	Utility function to initialize pseudo data loader
	'''
	def initialize_pseudo_trgt_dataloader(self,dom):
		calc_lbl = False
		if self.current_iteration >= max(self.settings['val_after'],self.settings['enough_iter']):
			calc_lbl = True

		if not calc_lbl:
			return

		len_indx_list = len(self.pseudo_indx_list[dom])
		rem_topK_beg  = int(len_indx_list*((1-self.curr_topK)/2))
		rem_topK_end  = len_indx_list - rem_topK_beg

		pseudo_target_dataset_train = PseudoTargetDataset('_'.join([dom,'train.npy']),self.pseudo_indx_list[dom][rem_topK_beg:rem_topK_end],self.pseudo_cls_label_list[dom][rem_topK_beg:rem_topK_end],self.pseudo_conf_list[dom][rem_topK_beg:rem_topK_end])
		self.pseudo_target_dl_iter_train_list[dom] = iter(DataLoader(pseudo_target_dataset_train, batch_size=self.batch_size,shuffle=False,drop_last=True))


	'''
	Utility function to set the model in eval or train mode
	'''
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

	'''
	Target dataset validation
	'''
	def val_over_target_set(self):
		self.get_all_val_target_dataloaders()
		with torch.no_grad():
			
			for dom in self.settings['trgt_datasets']:

				data_metric_trgt_summ  		     =  {key: utils.Summ(grp='_'.join(['metrics',dom,self.grp_name_suff]),key=key,func=self.settings['func']['curr_val'], writer=self.test_writer) for key in self.settings['data_metrics'] if self.settings['data_metrics'][key] == True}
				batch_metric_trgt_summ 		     =  {key: utils.Summ(grp='_'.join(['metrics',dom,self.grp_name_suff]),key=key,func=self.settings['func']['avg_val'], writer=self.test_writer) for key in self.settings['batch_metrics'] if self.settings['batch_metrics'][key] == True}
				key_acc_trgt_summ    	 	     =  {key: utils.Summ(grp='_'.join(['accuracy',dom,self.grp_name_suff]),key='_'.join([str(key),'acc']),func=self.settings['func']['curr_val'], writer=self.test_writer) for key in self.settings['calc_logits']}
				
				all_labels_trgt 				 =  []
				all_preds_trgt  				 =  []
				all_confs_trgt  				 =  []
				all_indices_trgt    			 =  []
				all_M_trgt						 =  []
				
				
				all_key_preds = {i:[]for i in self.settings['calc_logits']}
				target_dl_iter_val_list = self.target_dl_iter_val_list[dom]
				

				for data in tqdm(target_dl_iter_val_list,desc=dom):

					indx,images,label,_  		= data

					x				 			= images.to(self.settings['device']).float()
					label			 			= label.to(self.settings['device']).long()

					G 							= self.network.model['global']['G'](x)
					F 							= self.network.model['global']['Fs'](G)
					M 							= self.network.model['global']['M'](F)


					for key in self.settings['calc_logits']:
						cls_logits,_,mat 			= metrics.get_logits(key=key,feats={'M':M})
						_,cls_preds    		        = torch.max(cls_logits,dim=-1)
						all_key_preds[key].extend(list(cls_preds.cpu().numpy()))



					cls_logits,_,mat 			= metrics.get_logits(key=self.settings['target_label_logit_key'],feats={'M':M})
					cls_confs,cls_preds    		= torch.max(cls_logits,dim=-1)

					all_labels_trgt.extend(list(label.cpu().numpy()))
					all_preds_trgt.extend(list(cls_preds.cpu().numpy()))
					all_confs_trgt.extend(list(cls_confs.cpu().numpy()))
					all_indices_trgt.extend(list(indx.cpu().numpy()))
					all_M_trgt.append(M)
					

					for key in self.settings['batch_metrics']:
						if self.settings['batch_metrics'][key]:
							current_metric,count = metrics.get_metric(key, feats={'M':M,'matrix':mat,'cls_logits':cls_logits,'p':None,'cls_labels':label,'cls_preds':cls_preds,'cls_confs':cls_confs})
							batch_metric_trgt_summ[key].update(current_metric,count,self.current_iteration)

				all_M_trgt = torch.cat(all_M_trgt,dim=0).cpu().numpy()

				for key in self.settings['batch_metrics']:
					if self.settings['batch_metrics'][key]:
						batch_metric_trgt_summ[key].tb()


				for i in all_key_preds:
					cls_preds_i = all_key_preds[i]
					acc,count =  metrics.get_metric('cls_acc_data', feats={'all_labels':all_labels_trgt,'all_preds':cls_preds_i})
					key_acc_trgt_summ[i].update(acc,count,self.current_iteration)
					key_acc_trgt_summ[i].tb()


				for key in self.settings['data_metrics']:
					if self.settings['data_metrics'][key]:
						current_metric,count = metrics.get_metric(key, feats={'dom_name':dom,'all_labels':all_labels_trgt,'all_preds':all_preds_trgt,'all_confs':all_confs_trgt,'network':self.network})
						data_metric_trgt_summ[key].update(current_metric,count,self.current_iteration)
						data_metric_trgt_summ[key].tb()

				metrics.plot_acc_with_conf_thr(gt=all_labels_trgt,preds=all_preds_trgt,dom_name=dom,conf_preds=all_confs_trgt,writer=self.test_writer,curr_iterno=self.current_iteration)
				self.check_and_save_weights(data_metric_trgt_summ['cls_acc_data'].get_val(),dom)
				self.save_expt_data(gt=all_labels_trgt,preds=all_preds_trgt,conf_preds=all_confs_trgt,curr_cls_acc = data_metric_trgt_summ['cls_acc_data'].get_val(),all_M=all_M_trgt)

	'''
	Utility function to save experiment data
	'''
	def save_expt_data(self,gt,preds,conf_preds,curr_cls_acc,all_M):

		def get_ssm():
			gt_1 	   	   = np.array(gt)
			preds_1 	   = np.array(preds)
			conf_preds_1 = np.array(conf_preds)

			idx 	   = np.argsort(conf_preds_1)[::-1]
			bool_list  = (gt_1[idx] == preds_1[idx]).astype(float)
			L 		   = bool_list.shape[0]
			x 		   = np.arange(1,101,1)
			y 		   = [ np.mean(bool_list[:int((i/100.)*L)]) for i in x ]
			return x,y


		def get_ssm_agree():
			gt_1 	   	   = np.array(gt)
			preds_1 	   = np.array(preds)
			conf_preds_1 = np.array(conf_preds)

			K 		   = np.argmax(all_M,axis=-1)
			idx1 	   = (np.prod(np.float32(K == np.expand_dims(K[:,0],axis=-1)),axis=-1))== 1

			gt_1  	   = gt_1[idx1]
			preds_1    = preds_1[idx1]
			conf_preds_1 = conf_preds_1[idx1]

			idx 	   = np.argsort(conf_preds_1)[::-1]
			bool_list  = (gt_1[idx] == preds_1[idx]).astype(float)
			L 		   = bool_list.shape[0]
			x 		   = np.arange(1,101,1)
			y 		   = [ np.mean(bool_list[:int((i/100.)*L)]) for i in x ]

			return x,y


		def get_ssm_disagree():
			gt_1 	   	   = np.array(gt)
			preds_1 	   = np.array(preds)
			conf_preds_1 = np.array(conf_preds)

			K 		   = np.argmax(all_M,axis=-1)
			idx1 	   = (np.prod(np.float32(K == np.expand_dims(K[:,0],axis=-1)),axis=-1))== 0

			gt_1  	   = gt_1[idx1]
			preds_1    = preds_1[idx1]
			conf_preds_1 = conf_preds_1[idx1]

			idx 	   = np.argsort(conf_preds_1)[::-1]
			bool_list  = (gt_1[idx] == preds_1[idx]).astype(float)
			L 		   = bool_list.shape[0]
			x 		   = np.arange(1,101,1)
			y 		   = [ np.mean(bool_list[:int((i/100.)*L)]) for i in x ]

			return x,y


		self.max_test_acc = max(curr_cls_acc,self.max_test_acc)
		
		if self.max_test_acc == curr_cls_acc:
			self.expt_data['max_test_acc'] = self.max_test_acc
			self.expt_data['iterno']	   = self.current_iteration
		
		x_ssm,y_ssm = get_ssm()
		x_ssm_agree,y_ssm_agree = get_ssm_agree()
		x_ssm_dis,y_ssm_dis = get_ssm_disagree()


		if 'ssm_plots' not in self.expt_data:
			self.expt_data['ssm_plots'] = []

		if 'ssm_plots_agree' not in self.expt_data:
			self.expt_data['ssm_plots_agree'] = []

		if 'ssm_plots_disagree' not in self.expt_data:
			self.expt_data['ssm_plots_disagree'] = []

		self.expt_data['ssm_plots'].append((x_ssm,y_ssm))
		self.expt_data['ssm_plots_agree'].append((x_ssm_agree,y_ssm_agree))
		self.expt_data['ssm_plots_disagree'].append((x_ssm_dis,y_ssm_dis))

		all_M = torch.tensor(all_M)
		gt = torch.tensor(gt)
		preds= torch.tensor(preds)

		hf_acc_f_agree_recall,count1      =  metrics.conf_acc_f_aggreement(all_M,gt,preds)
		hf_acc_f_dis_agree_recall,count2  =  metrics.conf_acc_f_dis_aggreement(all_M,gt,preds)
		hf_acc_g_precision_agree,count3   =  metrics.conf_acc_g_agreement(all_M,gt,preds)
		hf_acc_g_precision_disagre,count4 =  metrics.conf_acc_g_dis_agreement(all_M,gt,preds)

		if 'hf_acc_f_agree_recall' not in self.expt_data:
			self.expt_data['hf_acc_f_agree_recall'] = ([],[])

		if 'hf_acc_f_dis_agree_recall' not in self.expt_data:
			self.expt_data['hf_acc_f_dis_agree_recall'] = ([],[])
		
		if 'hf_acc_g_precision_agree' not in self.expt_data:
			self.expt_data['hf_acc_g_precision_agree'] = ([],[])

		if 'hf_acc_g_precision_disagre' not in self.expt_data:
			self.expt_data['hf_acc_g_precision_disagre'] = ([],[])


		if count1 == 0:
			hf_acc_f_agree_recall = 0
		else:
			hf_acc_f_agree_recall = hf_acc_f_agree_recall.item()/count1.item()

		if count2 == 0:
			hf_acc_f_dis_agree_recall = 0
		else:
			hf_acc_f_dis_agree_recall = hf_acc_f_dis_agree_recall.item()/count2.item()

		if count3 == 0:
			hf_acc_g_precision_agree = 0
		else:
			hf_acc_g_precision_agree = hf_acc_g_precision_agree.item()/count3

		if count4 == 0:
			hf_acc_g_precision_disagre = 0
		else:
			hf_acc_g_precision_disagre = hf_acc_g_precision_disagre.item()/count4

		
		self.expt_data['hf_acc_f_agree_recall'][0].append(self.current_iteration)
		self.expt_data['hf_acc_f_agree_recall'][1].append(hf_acc_f_agree_recall)

		self.expt_data['hf_acc_f_dis_agree_recall'][0].append(self.current_iteration)
		self.expt_data['hf_acc_f_dis_agree_recall'][1].append(hf_acc_f_dis_agree_recall)

		self.expt_data['hf_acc_g_precision_agree'][0].append(self.current_iteration)
		self.expt_data['hf_acc_g_precision_agree'][1].append(hf_acc_g_precision_agree)

		self.expt_data['hf_acc_g_precision_disagre'][0].append(self.current_iteration)
		self.expt_data['hf_acc_g_precision_disagre'][1].append(hf_acc_g_precision_disagre)

		met = []
		#met.append(['hf_acc_f_agree_recall', hf_acc_f_agree_recall])
		#met.append(['hf_acc_f_dis_agree_recall', hf_acc_f_dis_agree_recall])
		#met.append(['hf_acc_g_precision_agree', hf_acc_g_precision_agree])
		#met.append(['hf_acc_g_precision_disagre', hf_acc_g_precision_disagre])
		met.append(['target_domain_accuracy',curr_cls_acc])
		logger.msg(tabulate(met,headers=['metrics','value'],tablefmt="simple",showindex=True))

		torch.save(self.expt_data, os.path.join(self.settings['weights_path'],self.exp_name,'expt_data.pth'))


	'''
	Function to calculate the loss value
	'''
	def get_loss(self,which_loss):

		train_loss_summ 		= utils.Summ(grp='train',key=which_loss,func=self.settings['func']['curr_val'], writer=self.train_writer)
		train_acc_summ 			= utils.Summ(grp='train',key='cls_acc',func=self.settings['func']['curr_val'], writer=self.train_writer)

		src_M 					= self.src_features['M']
		src_labels 				= self.src_features['label']
		src_dom_labels          = self.src_features['domain_label']
		
		if self.current_iteration > max(self.settings['val_after'],self.settings['enough_iter']):
			pseudo_target_M 		= self.pseudo_target_features['M']
			pseudo_target_labels    = self.pseudo_target_features['label']


		if which_loss == 'source':

			M                       = src_M
			labels 					= src_labels
			loss_1 					= metrics.l4_mirror_CE(src_M/self.settings['softmax_temperature'],src_labels)
			tot_loss 				= loss_1

		elif which_loss == 'target':

			M                       = src_M
			labels 					= src_labels

			if self.current_iteration > max(self.settings['val_after'],self.settings['enough_iter']):
				loss_2              = metrics.l4_mirror_CE(pseudo_target_M/self.settings['softmax_temperature'],pseudo_target_labels)
				tot_loss			= loss_2
			
		train_loss_summ.update(curr_val=tot_loss,count=1,iterno=self.current_iteration)
		train_loss_summ.tb()

		cls_logits,_,_ 			= metrics.get_logits(key=self.settings['target_label_logit_key'],feats={'M':M})
		cls_confs,cls_preds    	= torch.max(cls_logits,dim=-1)

		curr_cls_acc,count = metrics.get_metric('cls_acc',feats={'cls_labels':labels,'cls_preds':cls_preds})

		train_acc_summ.update(curr_cls_acc,count,self.current_iteration)
		train_acc_summ.tb()

		if self.current_iteration % self.settings['log_interval'] == 0 or self.current_iteration % self.settings['log_interval'] == 1:
			logger.msg('{} loss  ={} '.format(which_loss,tot_loss.cpu()))

		return tot_loss


	'''
	Function to select active losses
	'''
	def loss(self):
		active_losses   = self.active_losses
		current_loss    = self.current_iteration%len(active_losses)
		optim 			= self.optimizer_dict[active_losses[current_loss]]
		optim.zero_grad()
		loss 			= self.get_loss(active_losses[current_loss])
		loss.backward()
		optim.step()

	'''
	Function to implement the forward prop
	'''
	def forward(self):

		current_loss    = self.current_iteration%len(self.active_losses)

		if self.active_losses[current_loss] == 'target':
		 	self.set_mode(self.settings['mode']['val'])
		else:
		 	self.set_mode(self.settings['mode']['train'])

		self.src_features 								= {}
		self.pseudo_target_features 					= {}
	
		image_batch_concat 								= []
		label_batch_concat  							= []
		domain_label_batch_concat  						= []

		for dom in self.src_data :
			images,label,domain_label 			    	= self.src_data[dom]['images'],self.src_data[dom]['label'],self.src_data[dom]['domain_label']
			image_batch_concat.append(images)
			label_batch_concat.append(label)
			domain_label_batch_concat.append(domain_label)
		
		image_batch_concat 								= torch.cat(image_batch_concat,dim=0)
		label_batch_concat								= torch.cat(label_batch_concat,dim=0)
		domain_label_batch_concat						= torch.cat(domain_label_batch_concat,dim=0)


		feats_G 										= self.network.model['global']['G'](image_batch_concat)
		feats_F 										= self.network.model['global']['Fs'](feats_G)
		feats_M 										= self.network.model['global']['M'](feats_F)

		self.src_features['G'] 				=  feats_G
		self.src_features['F'] 				=  feats_F
		self.src_features['M'] 				=  feats_M
		self.src_features['label']			=  label_batch_concat
		self.src_features['domain_label']	=  domain_label_batch_concat


		

		if self.active_losses[current_loss] == 'source':
		 	self.set_mode(self.settings['mode']['val'])
		else:
		 	self.set_mode(self.settings['mode']['train'])


		if self.current_iteration > max(self.settings['val_after'],self.settings['enough_iter']):
			pseudo_target_image_batch_concat 				= []
			pseudo_target_label_batch_concat  				= []

			for dom in self.pseudo_target_data :
				images,label 								= self.pseudo_target_data[dom]['images'],self.pseudo_target_data[dom]['label']
				pseudo_target_image_batch_concat.append(images)
				pseudo_target_label_batch_concat.append(label)

			pseudo_target_image_batch_concat 				= torch.cat(pseudo_target_image_batch_concat,dim=0)
			pseudo_target_label_batch_concat				= torch.cat(pseudo_target_label_batch_concat,dim=0)

			feats_G 										= self.network.model['global']['G'](pseudo_target_image_batch_concat)
			feats_F 										= self.network.model['global']['Fs'](feats_G)
			feats_M 										= self.network.model['global']['M'](feats_F)

			self.pseudo_target_features['G'] 				=  feats_G
			self.pseudo_target_features['F'] 				=  feats_F
			self.pseudo_target_features['M'] 				=  feats_M
			self.pseudo_target_features['label']			=  pseudo_target_label_batch_concat
			

	'''
	Function for training the the data
	This function is called at every iteration
	'''
	def train (self):

		self.src_data={}

		current_loss    = self.current_iteration%len(self.active_losses)
		cond_1 = self.active_losses[current_loss] not in self.settings['losses_after_enough_iters']
		cond_2 = self.current_iteration <= max(self.settings['val_after'],self.settings['enough_iter'])
		
		if (cond_1 and cond_2) or (not cond_2):

			for dom in self.settings['src_datasets']:
				try:
					self.src_data[dom]={}
					_,self.src_data[dom]['images'],self.src_data[dom]['label'],self.src_data[dom]['domain_label'] = self.source_dl_iter_train_list[dom].next()
					self.src_data[dom]['images'] = Variable(self.src_data[dom]['images']).to(self.settings['device']).float()
					self.src_data[dom]['label'] = Variable(self.src_data[dom]['label']).to(self.settings['device']).long()
					self.src_data[dom]['domain_label'] = Variable(self.src_data[dom]['domain_label']).to(self.settings['device']).long()


				except StopIteration:
					self.initalize_src_train_dataloader(dom)
					self.src_data[dom]={}
					_,self.src_data[dom]['images'],self.src_data[dom]['label'],self.src_data[dom]['domain_label'] = self.source_dl_iter_train_list[dom].next()
					self.src_data[dom]['images'] = Variable(self.src_data[dom]['images']).to(self.settings['device']).float()
					self.src_data[dom]['label'] = Variable(self.src_data[dom]['label']).to(self.settings['device']).long()
					self.src_data[dom]['domain_label'] = Variable(self.src_data[dom]['domain_label']).to(self.settings['device']).long()


			if self.current_iteration > max(self.settings['val_after'],self.settings['enough_iter']):

				self.pseudo_target_data={}


				if self.current_iteration == max(self.settings['val_after'],self.settings['enough_iter'])+1:
					self.initialize_pseudo_target_indices()
					for dom in self.settings['trgt_datasets']:
						self.initialize_pseudo_trgt_dataloader(dom)

				for dom in self.settings['trgt_datasets']:

					try:
						self.pseudo_target_data[dom]={}
						_,self.pseudo_target_data[dom]['images'],self.pseudo_target_data[dom]['label'],_,_ =  self.pseudo_target_dl_iter_train_list[dom].next()
						self.pseudo_target_data[dom]['images'] = Variable(self.pseudo_target_data[dom]['images']).to(self.settings['device']).float()
						self.pseudo_target_data[dom]['label'] = Variable(self.pseudo_target_data[dom]['label']).to(self.settings['device']).long()

					except StopIteration:
						self.initialize_pseudo_target_indices()
						self.initialize_pseudo_trgt_dataloader(dom)
						self.pseudo_target_data[dom]={}
						_,self.pseudo_target_data[dom]['images'],self.pseudo_target_data[dom]['label'],_,_ = self.pseudo_target_dl_iter_train_list[dom].next()
						self.pseudo_target_data[dom]['images'] = Variable(self.pseudo_target_data[dom]['images']).to(self.settings['device']).float()
						self.pseudo_target_data[dom]['label'] = Variable(self.pseudo_target_data[dom]['label']).to(self.settings['device']).long()

			self.forward()
			self.loss()

if __name__=='__main__':
	raise NotImplementedError('Please check train file')

