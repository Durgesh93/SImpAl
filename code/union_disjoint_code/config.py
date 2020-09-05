import os
import shutil
import torch
import glob
import pdb
import numpy as np
from config_populate import data_settings


def get_best_iter(exp_name):
		all_paths  							= [x for x in os.listdir(os.path.join(settings['weights_path'],exp_name)) if (('expt_data' not in x) and ('enough_iter' not in x) )]
		best_iter 							= max(set(map(lambda x : int(x.split('.')[0].split('_')[1]),all_paths)))
		return best_iter


def gen_exp_name():
	st ='expt'
	st = '_'.join([st,str(settings['bb'])])
	st = '_'.join([st,dataset_name])
	st = '_'.join([st,data_key])

	if len(settings['optimizer_dict'] )>0:
		active_losses = [loss for loss in settings['optimizer_dict'] if settings['use_loss'][loss] == True ]
		if len(active_losses) == 1:
			loss_str = '_'.join(active_losses)
		else:
			loss_str = '_'.join(['alternate','_'.join(active_losses)])

		st = '_'.join([st,loss_str])

	if settings['recalc_pseudo_labels']:
		st = '_'.join([st,'recalc_pseudo_labels']) 

	st = '_'.join([st,'topK',str(settings['topK'])])
	
	st = '_'.join([st,'single_stage'])

	if len(settings['id_str'])>0:
		st = '_'.join([st,settings['id_str']])

	return st


def start_tb():
	from tensorboard import program
	tb = program.TensorBoard()
	if 'TB_BIND_ALL' in os.environ:
		is_bind_all = os.environ['TB_BIND_ALL']
	else:
		is_bind_all = 0

	if is_bind_all:
		tb.configure(argv=[None,'--port',str(settings['tb_port_no']), '--logdir',os.path.join(server_root_path,'summaries',settings['exp_name']),'--bind_all' ])
	else:
		tb.configure(argv=[None,'--port',str(settings['tb_port_no']), '--logdir',os.path.join(server_root_path,'summaries',settings['exp_name'])])
	url = tb.launch()



settings   								= {}

server_root_path  						= '../../'
dataset_name 							= 'office-31'
data_key 								= 'AD_W'

settings['server_root_path']		    = server_root_path
settings['dataset_dir'] 				= os.path.join('data',dataset_name)


settings['quite_mode']                  = False


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
settings['balance_dataset']			    = False



settings['bb']	 						= 'resnet50'
settings['bb_output'] 					= 2048	
settings['F_dims'] 						= 256


settings['to_train']					= {

											'global': {
													'G' : True,
													'Fs': True,
													'M' : True,
													}
											
										  }


settings['softmax_temperature']			= 1


settings['use_loss']    			    = {	
											'source':True,
											'target':True,
										  }

settings['losses_after_enough_iters']    = ['target']

#optimizer settings
settings['optimizer_dict'] 				= {	
											'source':{'global': ['G', 'Fs', 'M']},
											'target':{'global': ['G', 'Fs', 'M']}
										  }


settings['lr']				= 			  {
											'source':1e-5,
											'target':1e-5,
										  }



settings['calc_logits']					= [8]


def  avg_val_fn(curr_val,tot_val,curr_count,tot_count):
	if tot_count == 0:
		return 0
	else:
		return tot_val/tot_count

def curr_val_fn(curr_val,tot_val,curr_count,tot_count):
	if curr_count == 0:
		return 0
	else:
		return curr_val/curr_count


settings['func']						= {
											'curr_val': lambda curr_val,tot_val,curr_count,tot_count: curr_val_fn(curr_val,tot_val,curr_count,tot_count),
											'avg_val': lambda curr_val,tot_val,curr_count,tot_count:avg_val_fn(curr_val,tot_val,curr_count,tot_count),
										  }



settings['data_metrics']				= {
											'cls_acc_data'				 : True
										  }

    
settings['batch_metrics']		 		= {
											'cls_acc'  					 : True,
											'hf_acc_f_agree_recall' 	 : True,
											'hf_acc_f_dis_agree_recall'  : True,
											'hf_acc_g_precision_agree' 	 : True,
											'hf_acc_g_precision_disagre' : True,
											'classifier_agreement_metric': True,
											
										  }




settings['target_label_logit_key']     = 8



#training settings
settings['id_str']						= 'disjoint'
settings['recalc_pseudo_labels']        = True
settings['topK']                   		= 1
settings['domain-net_frac']				= 1
settings['exp_name'] 					= gen_exp_name()

settings['validate_target_before_enough_iters'] = True

settings['mode']						= {'train':0,'val':1}
settings['summaries_path'] 				= os.path.join(server_root_path, 'summaries')
settings['weights_path'] 				= os.path.join(server_root_path, 'weights')
  
settings['gpu'] 						= 0
settings['device'] 						= 'cuda:' + str(settings['gpu'])
torch.cuda.set_device(settings['gpu'])
settings['tb_port_no'] 					= 9999-int(settings['gpu'])

settings['checkpoints_count']			= 1

settings['expt_dict']					= {
											'office-31':{
															'AD_W':{'enough_iter':4000,'max_iter':10000,'val_after':200,'batch_size':16,'val_batch_size_factor':20},
															'DW_A':{'enough_iter':4000,'max_iter':10000,'val_after':200,'batch_size':16,'val_batch_size_factor':20},
															'AW_D':{'enough_iter':4000,'max_iter':10000,'val_after':200,'batch_size':16,'val_batch_size_factor':20}
														},
											'domain-net':{
															'CIPQR_S':{'enough_iter':120000,'max_iter':500000,'val_after':12000,'batch_size':8,'val_batch_size_factor':40},
															'CIPQS_R':{'enough_iter':120000,'max_iter':500000,'val_after':12000,'batch_size':8,'val_batch_size_factor':40},
															'CIPSR_Q':{'enough_iter':120000,'max_iter':500000,'val_after':12000,'batch_size':8,'val_batch_size_factor':40},
															'CPQRS_I':{'enough_iter':120000,'max_iter':500000,'val_after':12000,'batch_size':8,'val_batch_size_factor':40},
															'CIQRS_P':{'enough_iter':120000,'max_iter':500000,'val_after':12000,'batch_size':8,'val_batch_size_factor':40},
															'IPQRS_C':{'enough_iter':180000,'max_iter':500000,'val_after':12000,'batch_size':8,'val_batch_size_factor':40}
														},
											'image-clef':{
															'PC_I':{'enough_iter':1200,'max_iter':3000,'val_after':100,'batch_size':16,'val_batch_size_factor':20},
															'IC_P':{'enough_iter':1500,'max_iter':3000,'val_after':100,'batch_size':16,'val_batch_size_factor':20},
															'IP_C':{'enough_iter':1200,'max_iter':3000,'val_after':100,'batch_size':16,'val_batch_size_factor':20}
														},

											'office-home':{
															'ACP_R':{'enough_iter':15000,'max_iter':30000,'val_after':1000,'batch_size':16,'val_batch_size_factor':20},
															'ACR_P':{'enough_iter':15000,'max_iter':30000,'val_after':1000,'batch_size':16,'val_batch_size_factor':20},
															'APR_C':{'enough_iter':15000,'max_iter':30000,'val_after':1000,'batch_size':16,'val_batch_size_factor':20},
															'CPR_A':{'enough_iter':15000,'max_iter':30000,'val_after':1000,'batch_size':16,'val_batch_size_factor':20}
														},
											'office-caltech':{
															'ACD_W':{'enough_iter':1000,'max_iter':3000,'val_after':100,'batch_size':16,'val_batch_size_factor':10},
															'ADW_C':{'enough_iter':1000,'max_iter':3000,'val_after':100,'batch_size':16,'val_batch_size_factor':10},
															'ACW_D':{'enough_iter':1000,'max_iter':3000,'val_after':100,'batch_size':16,'val_batch_size_factor':10},
															'CDW_A':{'enough_iter':1000,'max_iter':3000,'val_after':100,'batch_size':16,'val_batch_size_factor':10}
														}
										  }

settings['log_interval']				= settings['expt_dict'][dataset_name][data_key]['val_after']
settings['start_iter'] 					= 0
settings['max_iter']					= settings['expt_dict'][dataset_name][data_key]['max_iter']
settings['enough_iter'] 				= settings['expt_dict'][dataset_name][data_key]['enough_iter']
settings['val_after']   				= settings['expt_dict'][dataset_name][data_key]['val_after']
settings['batch_size']  				= settings['expt_dict'][dataset_name][data_key]['batch_size']
settings['val_batch_size_factor']		= settings['expt_dict'][dataset_name][data_key]['val_batch_size_factor']
settings['load_model']					= False
settings['load_opt']					= False
settings['model_dict']					= {'exp_name':'exp1_test','iter':51}
settings['opt_dict']					= {'exp_name':'exp1_test','iter':51}


settings['continue_training']			= False

if settings['continue_training']:
	if not settings['quite_mode']:
		ip = raw_input('Saving continue training continue to save? (y/n): ')
		ip = str(ip)
		if ip.lower() == 'y' or str(ip.lower()) == 'yes':
			best_iter                           = get_best_iter(settings['exp_name'])
			settings['model_dict']['exp_name']  = settings['exp_name']
			settings['model_dict']['iter']  	= best_iter
			settings['opt_dict']['exp_name']	= settings['exp_name']
			settings['opt_dict']['iter']	    = best_iter
			settings['load_model']				= True
			settings['load_opt']				= True
			settings['start_iter']				= best_iter+1
		else:
			print('make settings[continue_training] to False to RUN')
			exit()
	else:
		best_iter                           = get_best_iter(settings['exp_name'])
		settings['model_dict']['exp_name']  = settings['exp_name']
		settings['model_dict']['iter']  	= best_iter
		settings['opt_dict']['exp_name']	= settings['exp_name']
		settings['opt_dict']['iter']	    = best_iter
		settings['load_model']				= True
		settings['load_opt']				= True
		settings['start_iter']				= best_iter+1





settings['save_code']				    = False

if settings['save_code']:
	if not settings['quite_mode']:
		ip = raw_input('Saving code mode continue to save? (y/n): ')
		ip = str(ip)

		if ip.lower() == 'y' or str(ip.lower()) == 'yes':
			pyc_files = glob.glob(os.path.join('.','*.pyc'))
			for a in pyc_files:
				os.remove(a)
			shutil.copytree(os.path.join('.'),os.path.join('..','archive','_'.join(['code',settings['exp_name']])))
			print('make settings[save_code] to False to RUN')
			exit()
		else:
			print('Decided not to execute!')
			print('make settings[save_code] to False to RUN')
			exit()
	else:
		pyc_files = glob.glob(os.path.join('.','*.pyc'))
		for a in pyc_files:
			os.remove(a)
		shutil.copytree(os.path.join('.'),os.path.join('..','archive','_'.join(['code',settings['exp_name']])))



