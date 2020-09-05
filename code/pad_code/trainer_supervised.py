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

from pdb import set_trace as bp

logger = logging.getLogger()

class TrainerG(object):

	def __init__(self):


		self.settings 									= config.settings
		self.network 									= mmodel().to(self.settings['device'])

		self.to_train 									= self.settings['to_train']
		self.writer										= SummaryWriter(os.path.join('summaries', self.settings['exp_name'],'logval'))

		self.val_batch_size 							= self.settings['val_batch_size']
		
		self.exp_name									= self.settings['exp_name']
		self.phase										= self.settings['mode']['val']

		self.source_dl_iter_val_list 					= {}
		self.target_dl_iter_val_list 					= {}


	def load_weights_model(self):

		load_weights_path = os.path.join(self.settings['server_root_path'],'weights',self.settings['model_dict']['exp_name'],'model_' + str(self.settings['model_dict']['iter']) + '.pth')
		dict_to_load = torch.load(load_weights_path,map_location=self.settings['device'])
		model_state_dict = dict_to_load['model_state_dict']

		for nc,compts in self.network.model.items():
			for name,comp in compts.items():
				self.network.model[nc][name].load_state_dict(model_state_dict['_'.join([nc,name])])
		

	def  get_all_val_src_dataloaders(self):
		for dom in self.settings['src_datasets']:
			self.initalize_src_val_dataloader(dom)

	def  get_all_val_target_dataloaders(self):
		for dom in self.settings['trgt_datasets']:
			self.initalize_target_val_dataloader(dom)

	def initalize_src_val_dataloader(self,dom):
		source_dataset_val = TemplateDataset('_'.join([dom,'train.npy']),aug=False)
		self.source_dl_iter_val_list[dom] = iter(DataLoader(source_dataset_val, batch_size=self.val_batch_size, shuffle=False, num_workers=5,drop_last=False,pin_memory=True))



	def initalize_target_val_dataloader(self,dom):
		target_dataset_val = TemplateDataset('_'.join([dom,'val.npy']),aug=False)	
		self.target_dl_iter_val_list[dom] = iter(DataLoader(target_dataset_val, batch_size=self.val_batch_size, shuffle=False,drop_last=False, num_workers=5,pin_memory=True))
	


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





	def pad_distance(self):
		self.get_all_val_target_dataloaders()
		self.get_all_val_src_dataloaders()

		feats = []
		lab = []
		clslab = []


		with torch.no_grad():
			for dom in self.settings['src_datasets']:
				for data in tqdm(self.source_dl_iter_val_list[dom],desc=dom):
					indx,images,label,domain_lbl= data

					x				 			= images.to(self.settings['device']).float()
					label			 			= label.to(self.settings['device']).long()
					domain_lbl 					= domain_lbl.to(self.settings['device']).long()
		
					G 							= self.network.model['global']['G'](x)
					F							= self.network.model['global']['Fs'](G)
					M 							= self.network.model['global']['M'](F)

					cls_logits,_,mat 			= metrics.get_logits(key=self.settings['target_label_logit_key'],feats={'M':M})
					cls_confs,cls_preds    		= torch.max(cls_logits,dim=-1)
					feats.append(F)
					lab.extend([self.settings['src_datasets'].index(dom)] * F.shape[0])
					clslab.append(label)


			tar_label = len(self.settings['src_datasets'])
			for dom in self.settings['trgt_datasets']:
				for data in tqdm(self.target_dl_iter_val_list[dom],desc=dom):

					indx,images,label,_  		= data

					x				 			= images.to(self.settings['device']).float()
					label			 			= label.to(self.settings['device']).long()

					G 							= self.network.model['global']['G'](x)
					F							= self.network.model['global']['Fs'](G)
					M 							= self.network.model['global']['M'](F)
					cls_logits,_,mat 			= metrics.get_logits(key=self.settings['target_label_logit_key'],feats={'M':M})
					cls_confs,cls_preds    		= torch.max(cls_logits,dim=-1)

					feats.append(F)
					lab.extend([tar_label] * F.shape[0])
					clslab.append(label)
			
			feats = torch.cat(feats)
			lab = torch.from_numpy(np.array(lab))
			clslab = torch.cat(clslab)

			num_classes = M.shape[2]
			
			print(feats.shape, lab.shape, clslab.shape, num_classes)

			
			#Class-wise PAD
			##### Step 1 get nC2 
			# from itertools import product
			# nc2 = product(range(tar_label + 1), range(tar_label + 1)) # tarlabel is len of 
			# nc2 = [(i, j) for (i, j) in nc2 if i < j]

			# for pair in nc2:
			# 	print('Pair: {}'.format(pair))
			# 	res = []
			# 	for clss in range(num_classes):
			# 		feats_clss = feats[clslab == clss]
			# 		dom_lab = lab[clslab == clss]

			# 		dom1 = pair[0]
			# 		dom2 = pair[1]

			# 		feats_clss = torch.cat([feats_clss[dom_lab == dom1], feats_clss[dom_lab == dom2]], 0)
			# 		dom_lab = torch.cat([dom_lab[dom_lab == dom1], dom_lab[dom_lab == dom2]], 0)
			# 		# print(clss, feats_clss.shape, dom_lab.shape)
			# 		res.append(self.class_pad_metric(feats_clss, dom_lab))

			# 	print('{}'.format(' '.join([str(float(i)) for i in np.around(res, 2)])))

			# domain PAD
			from itertools import product
			nc2 = product(range(tar_label + 1), range(tar_label + 1)) # tarlabel is len of 
			nc2 = [(i, j) for (i, j) in nc2 if i < j]

			for pair in nc2:
				# print('Pair: {}'.format(pair))
				dom1, dom2 = pair[0], pair[1]
				_feats = torch.cat([feats[lab == dom1], feats[lab == dom2]], 0)
				_dom_lab = torch.cat([lab[lab == dom1], lab[lab == dom2]], 0)
				# print(clss, feats_clss.shape, dom_lab.shape)
				res = self.class_pad_metric(_feats, _dom_lab)

				print('Pair {}: {}'.format(pair, np.around(res, 2)))

			# bp()



	def class_pad_metric(self, feats, domain_labs):
		from sklearn.linear_model import SGDClassifier
		train_val_split = 0.8
		train_samples_num = int(train_val_split * feats.shape[0])
		# print(feats.shape, domain_labs.shape)
		npy = lambda x: x.detach().cpu().numpy()

		# 25 retries
		error = []
		for i in range(25):
			idx = np.arange(feats.shape[0])
			np.random.shuffle(idx)
			
			_feats = npy(feats[idx])
			_domain_labs = npy(domain_labs[idx])

			train_feats = _feats[:train_samples_num]
			train_labs = _domain_labs[:train_samples_num]

			val_feats = _feats[train_samples_num:]
			val_labs = _domain_labs[train_samples_num:]
			model = SGDClassifier(loss='modified_huber')
			model.fit(train_feats, train_labs)
			preds = model.predict(val_feats)
			acc = np.sum(preds == val_labs) * 1.0 / val_feats.shape[0]
			e = 1 - acc
			error.append(2 * (1 - 2 * e))

		# print('error ', error)
		return np.average(error)





if __name__=='__main__':
	raise NotImplementedError('Please check train file')
