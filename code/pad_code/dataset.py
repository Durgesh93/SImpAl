import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch
import os
import random
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from utils import get_domain_mapping
import config as config

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
		height, width = mat.shape[:2] # image shape has 3 dimensions
		image_center = (width/2, height/2) # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape

		rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)

		# rotation calculates the cos and sin, taking absolutes of those.
		abs_cos = abs(rotation_mat[0,0])
		abs_sin = abs(rotation_mat[0,1])

		# find the new width and height bounds
		bound_w = int(height * abs_sin + width * abs_cos)
		bound_h = int(height * abs_cos + width * abs_sin)

		# subtract old image center (bringing image back to origo) and adding the new image center coordinates
		rotation_mat[0, 2] += bound_w/2 - image_center[0]
		rotation_mat[1, 2] += bound_h/2 - image_center[1]

		# rotate image with the new bounds and translated rotation matrix
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





class TemplateDataset(Dataset):

	def __init__(self,index_file_name,aug=True):

		self.dataset_dir              = config.settings['dataset_dir']
		self.resolution               = config.settings['resolution']
		self.server_root_path         = config.settings['server_root_path']
		dom_mapping                   = get_domain_mapping(config.settings['src_datasets'],config.settings['trgt_datasets'])

		for dom in dom_mapping:
			if dom in index_file_name:
				self.domain_label     = dom_mapping[dom]

		self.index_list               = np.load(os.path.join(self.server_root_path,self.dataset_dir,config.settings['model_dict']['exp_name'],config.settings['index_list'],index_file_name),allow_pickle=True)

		self.transforms               = {   'R1':RotateImage(-15),'R2':RotateImage(-10),'R3':RotateImage(-5),'R4':RotateImage(5),'R5':RotateImage(10),'R6':RotateImage(15),
											'F':transforms.Compose([transforms.RandomHorizontalFlip(p=0.5)]),
											'FC':RGBFlip(),
											'J': transforms.ColorJitter(brightness=0.25, contrast=0.40, saturation=0.30, hue=0.50),
											'T':transforms.ToTensor()
										}

		self.AUG_TYPES                = ['I','R1','R2','R3','R4','R5','R6','F','FC','J']

		if aug:
			self.aug_list             = [[indx,x[0],int(x[1]),y,int(self.domain_label)] for y in self.AUG_TYPES for indx, x in enumerate(self.index_list)]
		else:
			self.aug_list             = [[indx,x[0],int(x[1]),'I',int(self.domain_label)] for indx, x in enumerate(self.index_list)]

	def __len__(self):
		return len(self.aug_list)

	def __getitem__(self, idx):

		indx,img_path,cat,aug_type,domain_label = self.aug_list[idx]

		#reading the image
		img = Image.open(img_path)
		if img.mode != 'RGB':
			img = img.convert(mode='RGB')

		#resizing to desired resolution
		img = img.resize((self.resolution,self.resolution))
		#transformations
		if aug_type == 'I':
			img = img
		else:
			img = self.transforms[aug_type](img)

		img = self.transforms['T'](img)
		return indx,img,cat,domain_label

