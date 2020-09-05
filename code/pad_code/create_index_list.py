import os
import sys
import numpy as np
import glob
import random
import config as config
from  utils import get_catg_mapping


def prepare_indices_generic(config):

	C 					= config.settings['C']
	C_dash 				= config.settings['C_dash']

	server_root_path    = config.server_root_path
	dataset_dir 		= config.settings['dataset_dir']

	src_datasets  		= config.settings['src_datasets']
	trgt_datasets  		= config.settings['trgt_datasets']
	resolution 			= config.settings['resolution']


	if not os.path.exists(os.path.join(server_root_path,dataset_dir, config.settings['exp_name'])):
		os.mkdir(os.path.join(server_root_path, dataset_dir,config.settings['exp_name']))


	if not os.path.exists(os.path.join(server_root_path,dataset_dir, config.settings['exp_name'], config.settings['index_list'])):
		os.mkdir(os.path.join(server_root_path,dataset_dir, config.settings['exp_name'], config.settings['index_list']))


	#using full source dataset for training like MFSAN
	for dataset_name in src_datasets:
		cat_dict    = {}
		imgs_train_paths = []
		print('creating index_list for {}'.format(dataset_name))
		cat_mapping = get_catg_mapping(C[dataset_name],C_dash[dataset_name])

		for catg in  os.listdir(os.path.join(server_root_path, dataset_dir, dataset_name)):
			if catg in C[dataset_name] or catg in C_dash[dataset_name]:
				cat_id 	= cat_mapping[catg]
				imgs_train_paths.extend([[x,cat_id] for x in glob.glob(os.path.join(server_root_path, dataset_dir,dataset_name,catg,'*'))])
			
		save_path = os.path.join(server_root_path, dataset_dir,config.settings['exp_name'], config.settings['index_list'], '_'.join([dataset_name,'train.npy']))
		np.save(save_path,imgs_train_paths)



	shared_trgt_catgs =  set([j for x in src_datasets for j in C[x]])


	#using full target dataset for pseudo labels generation and same target dataset for validation
	for dataset_name in trgt_datasets:

		imgs_paths = []
	
		cat_mapping = get_catg_mapping(shared_trgt_catgs,C_dash[dataset_name])
		print('creating index_list for target {}'.format(dataset_name))

		for catg in  os.listdir(os.path.join(server_root_path, dataset_dir, dataset_name)):
			if catg in shared_trgt_catgs or catg in C_dash[dataset_name]:
				cat_id 	= cat_mapping[catg]
				imgs_paths.extend([[x,cat_id] for x in glob.glob(os.path.join(server_root_path, dataset_dir, dataset_name,catg,'*'))])
		
		save_path = os.path.join(server_root_path, dataset_dir,config.settings['exp_name'], config.settings['index_list'], '_'.join([dataset_name,'train.npy']))
		np.save(save_path,imgs_paths)
		save_path = os.path.join(server_root_path, dataset_dir,config.settings['exp_name'], config.settings['index_list'], '_'.join([dataset_name,'val.npy']))
		print('Saving at path {}'.format(save_path))
		np.save(save_path,imgs_paths)


prepare_indices_generic(config)