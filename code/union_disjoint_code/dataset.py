import os
import cv2
import torch
import random
import numpy as np
from tqdm import tqdm
from PIL import Image
import config as config
import matplotlib.pyplot as plt
from utils import get_domain_mapping
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader



class RGBFlip(object):

	def __init__(self):
		pass

	def __call__(self, sample):
		sample = np.array(sample)
		newimg = np.copy(sample)
		reorder = np.arange(3)
		np.random.shuffle(reorder)
		newimg[:, :, 0] = sample[:, :, reorder[0]]
		newimg[:, :, 1] = sample[:, :, reorder[1]]
		newimg[:, :, 2] = sample[:, :, reorder[2]]
		newimg = Image.fromarray(newimg)
		return newimg


class RotateImage(object):

	def __init__(self,angle):
		self.chop_distances = {}
		self.angle = angle

	def rotateImage(self,mat, angle):
		height, width = mat.shape[:2]   
		image_center = (width/2, height/2) 

		rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)

		abs_cos = abs(rotation_mat[0,0])
		abs_sin = abs(rotation_mat[0,1])

		bound_w = int(height * abs_sin + width * abs_cos)
		bound_h = int(height * abs_cos + width * abs_sin)

		rotation_mat[0, 2] += bound_w/2 - image_center[0]
		rotation_mat[1, 2] += bound_h/2 - image_center[1]

		rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h))

		H, W, c = rotated_mat.shape

		d = self.get_chop_distance(rotated_mat, angle)
		chopped_image = rotated_mat[d : H-d, d : W-d]
		chopped_image[chopped_image == 1] =0
		resized_image = cv2.resize(chopped_image, (config.settings['resolution'], config.settings['resolution']))

		return resized_image

	def get_chop_distance(self,rotated_mat, angle):
		
		if config.dataset_name == 'office-31':
			if angle in self.chop_distances.keys():
				return self.chop_distances[angle]

			if angle > 0:

				x = 0
				y = 0

				while(rotated_mat[y, x, 0] == 0):
					y += 1

				self.chop_distances[angle] = y
				return y

			else:

				x = rotated_mat.shape[1] - 1
				y = 0

				while(rotated_mat[y, x, 0] == 0):
					y += 1
				self.chop_distances[angle] = y
				return y
		else:
			if angle in self.chop_distances.keys():
				return self.chop_distances[angle]

			if angle > 0:

				x = 0
				y = 0

				while(rotated_mat[y, x, 0] == 1):
					y += 1

				self.chop_distances[angle] = y
				return y

			else:

				x = rotated_mat.shape[1] - 1
				y = 0

				while(rotated_mat[y, x, 0] == 1):
					y += 1
				self.chop_distances[angle] = y
				return y


	def __call__(self, sample):
		sample = np.array(sample)
		newImg = self.rotateImage(sample,self.angle)
		newImg = Image.fromarray(newImg)
		return newImg





class frozen:

	def __init__(self, a):
		self.a = a

	def get(self):
		return self.a

	def __repr__(self):
		return str(self.a)

	def __str__(self):
		return self.a




class TemplateDataset(Dataset):

	def __init__(self,index_file_name,aug=True):

		self.dataset_dir              = config.settings['dataset_dir']
		self.resolution               = config.settings['resolution']
		self.server_root_path         = config.settings['server_root_path']
		dom_mapping                   = get_domain_mapping(config.settings['src_datasets'],config.settings['trgt_datasets'])

		
		self.aug = aug

		for dom in dom_mapping:
			if dom in index_file_name:
				self.domain_label     = dom_mapping[dom]

		self.index_list               = np.load(os.path.join(self.server_root_path,self.dataset_dir,config.settings['exp_name'],config.settings['index_list'],index_file_name),allow_pickle=True)

		self.transforms               = {   'R1':RotateImage(-15),'R2':RotateImage(-10),'R3':RotateImage(-5),'R4':RotateImage(5),'R5':RotateImage(10),'R6':RotateImage(15),
											'F':transforms.Compose([transforms.RandomHorizontalFlip(p=0.5)]),
											'FC':RGBFlip(),
											'C':transforms.Compose([transforms.Resize((256,256)), transforms.RandomCrop((224,224))]),
											'J': transforms.ColorJitter(brightness=0.25, contrast=0.40, saturation=0.30, hue=0.50),
											'T':transforms.ToTensor()
										}

		self.AUG_TYPES                = ['I', 'R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'F', 'FC', 'J', 'C', 'C', 'C', 'C', 'C']

		if aug:

			self.aug_list = [[ frozen( [indx,x[0],int(x[1]),y,int(self.domain_label)] ) for indx,x in enumerate(self.index_list) ]  for y in self.AUG_TYPES ]
			N = len(self.AUG_TYPES)
			self.aug_list = np.array(self.aug_list).reshape((N, -1))
			nA, nI = self.aug_list.shape
			for i in range(nI):
				np.random.shuffle(self.aug_list[:,i])
			self.aug_list = self.aug_list.flatten()

		else:
			self.aug_list             = [frozen([indx,x[0],int(x[1]),'I',int(self.domain_label)]) for indx, x in enumerate(self.index_list)]

		self.val_transform = transforms.Compose([transforms.Resize([224,224]), transforms.ToTensor()])

	def __len__(self):
		return len(self.aug_list)

	def __getitem__(self, idx):

		indx,img_path,cat,aug_type,domain_label = self.aug_list[idx].get()
		img = Image.open(img_path)

		if img.mode != 'RGB':
			img = img.convert(mode='RGB')

		img = img.resize((config.settings['resolution'],config.settings['resolution']))

		if self.aug:
			if aug_type!='I':
				img = self.transforms[aug_type](img)
			img = transforms.ToTensor()(img)
		else:
			img = self.val_transform(img)

		return indx,img,cat,domain_label


class PseudoTargetDataset(Dataset):

	def __init__(self,index_file_name,index,pseudo_gt,pseudo_conf):
		self.dataset_dir              = config.settings['dataset_dir']
		self.resolution               = config.settings['resolution']
		self.server_root_path         = config.settings['server_root_path']

		dom_mapping 				  = get_domain_mapping(config.settings['src_datasets'],config.settings['trgt_datasets'])
		self.aug = True

		for dom in dom_mapping:
			if dom in index_file_name:
				self.domain_label 	  = dom_mapping[dom]

		self.index_list               = np.load(os.path.join(self.server_root_path,self.dataset_dir,config.settings['exp_name'],config.settings['index_list'],index_file_name))
		self.index_list               = self.index_list[index]


		self.transforms               = {   'R1':RotateImage(-15),'R2':RotateImage(-10),'R3':RotateImage(-5),'R4':RotateImage(5),'R5':RotateImage(10),'R6':RotateImage(15),
											'F':transforms.Compose([transforms.RandomHorizontalFlip(p=0.5)]),
											'FC':RGBFlip(),
											'C':transforms.Compose([transforms.Resize((256,256)), transforms.RandomCrop((224,224))]),
											'J': transforms.ColorJitter(brightness=0.25, contrast=0.40, saturation=0.30, hue=0.50),
											'T':transforms.ToTensor()
										}

		self.AUG_TYPES                = ['I', 'R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'F', 'FC', 'J', 'C', 'C', 'C', 'C', 'C']



		self.aug_list = [[ frozen( [index[i],x[0],int(pseudo_gt[i]),pseudo_conf[i],y,int(self.domain_label)] ) for i,x in enumerate(self.index_list) ]  for y in self.AUG_TYPES ]
		
		N = len(self.AUG_TYPES)
		self.aug_list = np.array(self.aug_list).reshape((N, -1))
		nA, nI = self.aug_list.shape
		for i in range(nI):
				np.random.shuffle(self.aug_list[:,i])
		self.aug_list = self.aug_list.flatten()

		self.val_transform = transforms.Compose([transforms.Resize([224,224]), transforms.ToTensor()])


	def __len__(self):
		return len(self.aug_list)

	def __getitem__(self, idx):

		indx,img_path,cat,conf,aug_type,domain_label = self.aug_list[idx].get()

		img = Image.open(img_path)
		if img.mode != 'RGB':
			img = img.convert(mode='RGB')

		img = img.resize((config.settings['resolution'],config.settings['resolution']))
		
		if self.aug:
			if aug_type!= 'I':
				img = self.transforms[aug_type](img)
			img = transforms.ToTensor()(img)
		else:
			img = self.val_transform(img)

		return indx,img,cat,conf,domain_label