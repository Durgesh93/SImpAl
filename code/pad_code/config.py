import os
import shutil
import torch
import glob
from config_populate import data_settings
import numpy as np
import pdb
import sys

torch.manual_seed(2789863717929462893)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
np.random.seed(2570966908)



def start_tb():
	from tensorboard import program
	tb = program.TensorBoard()
	if 'TB_BIND_ALL' in os.environ:
		is_bind_all = os.environ['TB_BIND_ALL']
	else:
		is_bind_all = 0

	if is_bind_all:
		tb.configure(argv=[None,'--port',str(settings['tb_port_no']), '--logdir',os.path.join('summaries',settings['exp_name']),'--bind_all' ])
	else:
		tb.configure(argv=[None,'--port',str(settings['tb_port_no']), '--logdir',os.path.join('summaries',settings['exp_name'])])
	url = tb.launch()



settings   								= {}


server_root_path  						= '../../'
iters_from_argv 						= 2700
exp_name								= 'expt_resnet101_office-caltech_ADW_C_alternate_target_source_recalc_pseudo_labels_topK_1_single_stage_nav_full_psed_dl_newaug_run1'
settings['model_dict']					= {'exp_name':exp_name,'iter':iters_from_argv}
dataset_name 							= exp_name.split('_')[2]
data_key 								= '_'.join([exp_name.split('_')[3], exp_name.split('_')[4]])
settings['exp_name']					= exp_name #'DW_A_gt_pad_dist_'+str(settings['model_dict']['iter'])



settings['server_root_path']		    = server_root_path
settings['dataset_dir'] 				= os.path.join('data',dataset_name)



#dataset settings

settings['C'] 						    = data_settings[dataset_name][data_key]['C']
settings['C_dash'] 					    = data_settings[dataset_name][data_key]['C_dash']
settings['num_C_dash'] 				    = data_settings[dataset_name][data_key]['num_C_dash']

settings['num_C'] 					 	= data_settings[dataset_name][data_key]['num_C']
settings['src_datasets']        		= data_settings[dataset_name][data_key]['src_datasets']
settings['trgt_datasets']       		= data_settings[dataset_name][data_key]['trgt_datasets']

st0 = np.random.get_state()[1][0]
t0  = torch.initial_seed()

settings['seed_value']                  = {'torch':t0,'np':st0}
settings['resolution'] 					= 224

settings['index_list']					= 'index_list'

settings['bb']	 						= exp_name.split('_')[1]
settings['bb_output'] 					= 2048	
settings['F_dims'] 						= 256
settings['summaries_path'] 				= 'summaries'

settings['target_label_logit_key']      = 8
settings['to_train']					= {

											'global': {
													'G' : True,
													'Fs': True,
													'M' : True,
													}

										  }


settings['softmax_temperature']			= 1

settings['mode']						= {'train':0,'val':1}

settings['gpu'] 						= 1
settings['device'] 						= 'cuda:' + str(settings['gpu'])
torch.cuda.set_device(settings['gpu'])

settings['tb_port_no'] 					= 9999-int(settings['gpu'])
settings['val_batch_size']  			= 32


