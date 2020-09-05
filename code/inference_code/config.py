import os
import shutil
import torch
import glob
from config_populate import data_settings
import numpy as np



def get_best_iter(exp_name):
	all_paths  							= [x for x in os.listdir(os.path.join(settings['weights_path'],exp_name)) if (('expt_data' not in x) and ('enough_iter' not in x) )]
	best_iter 							= max(set(map(lambda x : int(x.split('.')[0].split('_')[1]),all_paths)))
	return best_iter



weights_dict 							= {
	
											'domain-net':{


														  'resnet101':
																	  {
																		'CIPQR_S':'expt_resnet101_domain-net_CIPQR_S_alternate_target_source_recalc_pseudo_labels_topK_1_single_stage_nav_full_psed_dl_newaug_run1',
																		'CIPQS_R':'expt_resnet101_domain-net_CIPQS_R_alternate_target_source_recalc_pseudo_labels_topK_1_single_stage_nav_full_psed_dl_newaug_run1',
																		'CIPSR_Q':'expt_resnet101_domain-net_CIPSR_Q_alternate_target_source_recalc_pseudo_labels_topK_1_single_stage_nav_full_psed_dl_newaug_run1',
																		'CIQRS_P':'expt_resnet101_domain-net_CIQRS_P_alternate_target_source_recalc_pseudo_labels_topK_1_single_stage_nav_full_psed_dl_newaug_run1',
																		'CPQRS_I':'expt_resnet101_domain-net_CPQRS_I_alternate_target_source_recalc_pseudo_labels_topK_1_single_stage_nav_full_psed_dl_newaug_run1'
																	  }


														 },

											'office-31':{
														  'resnet50':
																	{

																		'AW_D':'expt_resnet50_office-31_AW_D_alternate_target_source_recalc_pseudo_labels_topK_1_single_stage_nav_full_psed_dl_newaug_run2',
																		'DW_A':'expt_resnet50_office-31_DW_A_alternate_target_source_recalc_pseudo_labels_topK_1_single_stage_nav_full_psed_dl_newaug_run2',
																		'AD_W':'expt_resnet50_office-31_AD_W_alternate_target_source_recalc_pseudo_labels_topK_1_single_stage_nav_full_psed_dl_newaug_run2'
																	},

														   'resnet101':
														   			{
																		'AD_W':'expt_resnet101_office-31_AD_W_alternate_target_source_recalc_pseudo_labels_topK_1_single_stage_nav_full_psed_dl_newaug_run2',
																		'AW_D':'expt_resnet101_office-31_AW_D_alternate_target_source_recalc_pseudo_labels_topK_1_single_stage_nav_full_psed_dl_newaug_run2',
																		'DW_A':'expt_resnet101_office-31_DW_A_alternate_target_source_recalc_pseudo_labels_topK_1_single_stage_nav_full_psed_dl_newaug_run2',
														   			}
														},

											'image-clef':{
														  'resnet50':
																	{

																	'IC_P':'expt_resnet50_image-clef_IC_P_alternate_target_source_recalc_pseudo_labels_topK_1_single_stage_test_nav_full_psed_dl_newaug_run2',
																	'IP_C':'expt_resnet50_image-clef_IP_C_alternate_target_source_recalc_pseudo_labels_topK_1_single_stage_test_nav_full_psed_dl_newaug_run2',
																	'PC_I':'expt_resnet50_image-clef_PC_I_alternate_target_source_recalc_pseudo_labels_topK_1_single_stage_test_nav_full_psed_dl_newaug_run2',

																	},

														   'resnet101':
														   			{

																	'IC_P':'expt_resnet101_image-clef_IC_P_alternate_target_source_recalc_pseudo_labels_topK_1_single_stage_test_nav_full_psed_dl_newaug_run2',
																	'IP_C':'expt_resnet101_image-clef_IP_C_alternate_target_source_recalc_pseudo_labels_topK_1_single_stage_test_nav_full_psed_dl_newaug_run2',
																	'PC_I':'expt_resnet101_image-clef_PC_I_alternate_target_source_recalc_pseudo_labels_topK_1_single_stage_test_nav_full_psed_dl_newaug_run2',
																	
																	}
														},

											'office-home':{

														  'resnet101':
																	{

																	'ACP_R':'expt_resnet101_office-home_ACP_R_alternate_target_source_recalc_pseudo_labels_topK_1_single_stage_nav_full_psed_dl_newaug_run2',
																	'ACR_P':'expt_resnet101_office-home_ACR_P_alternate_target_source_recalc_pseudo_labels_topK_1_single_stage_nav_full_psed_dl_newaug_run2',
																	'APR_C':'expt_resnet101_office-home_APR_C_alternate_target_source_recalc_pseudo_labels_topK_1_single_stage_nav_full_psed_dl_newaug_run2',
																	'CPR_A':'expt_resnet101_office-home_CPR_A_alternate_target_source_recalc_pseudo_labels_topK_1_single_stage_nav_full_psed_dl_newaug_run2',
																	
																	},

														   'resnet50':
														   			{

																	'ACP_R':'expt_resnet50_office-home_ACP_R_alternate_target_source_recalc_pseudo_labels_topK_1_single_stage_nav_full_psed_dl_newaug_run2',
																	'ACR_P':'expt_resnet50_office-home_ACR_P_alternate_target_source_recalc_pseudo_labels_topK_1_single_stage_nav_full_psed_dl_newaug_run2',
																	'APR_C':'expt_resnet50_office-home_APR_C_alternate_target_source_recalc_pseudo_labels_topK_1_single_stage_nav_full_psed_dl_newaug_run2',
																	'CPR_A':'expt_resnet50_office-home_CPR_A_alternate_target_source_recalc_pseudo_labels_topK_1_single_stage_nav_full_psed_dl_newaug_run2',
														   			
														   			}
														},
											'office-caltech':{


														  'resnet101':
																	{															
																		'ACD_W':'expt_resnet101_office-caltech_ACD_W_alternate_target_source_recalc_pseudo_labels_topK_1_single_stage_nav_full_psed_dl_newaug_run2',
																		'ACW_D':'expt_resnet101_office-caltech_ACW_D_alternate_target_source_recalc_pseudo_labels_topK_1_single_stage_nav_full_psed_dl_newaug_run2',
																		'ADW_C':'expt_resnet101_office-caltech_ADW_C_alternate_target_source_recalc_pseudo_labels_topK_1_single_stage_nav_full_psed_dl_newaug_run2',
																		'CDW_A':'expt_resnet101_office-caltech_CDW_A_alternate_target_source_recalc_pseudo_labels_topK_1_single_stage_nav_full_psed_dl_newaug_run2',
																		
																	},
														  'resnet50':
														  			{
																		'ACD_W':'expt_resnet50_office-caltech_ACD_W_alternate_target_source_recalc_pseudo_labels_topK_1_single_stage_nav_full_psed_dl_newaug_run2',
																		'ACW_D':'expt_resnet50_office-caltech_ACW_D_alternate_target_source_recalc_pseudo_labels_topK_1_single_stage_nav_full_psed_dl_newaug_run2',
																		'ADW_C':'expt_resnet50_office-caltech_ADW_C_alternate_target_source_recalc_pseudo_labels_topK_1_single_stage_nav_full_psed_dl_newaug_run2',
																		'CDW_A':'expt_resnet50_office-caltech_CDW_A_alternate_target_source_recalc_pseudo_labels_topK_1_single_stage_nav_full_psed_dl_newaug_run2',
																	}													

														}
										  
										  }		
											
											
										  

settings   								= {}
server_root_path  						= '../../'
dataset_name 							= 'office-31'
data_key 								= 'DW_A'
settings['bb']	 						= 'resnet50'
settings['weights_path'] 				= os.path.join(server_root_path, 'weights')

settings['exp_name']                    = weights_dict[dataset_name][settings['bb']][data_key]
settings['iterno']						= get_best_iter(settings['exp_name'])
settings['server_root_path']		    = server_root_path
settings['dataset_dir'] 				= os.path.join('data',dataset_name)
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



settings['bb_output'] 					= 2048	
settings['F_dims'] 						= 256
settings['softmax_temperature']			= 1

settings['to_train']					= {

											'global': {
													'G' : True,
													'Fs': True,
													'M' : True,
													}
											
										  }


settings['target_label_logit_key']     = 8



#training settings

settings['mode']						= {'train':0,'val':1}


settings['gpu'] 						=  0
settings['device'] 						= 'cuda:' + str(settings['gpu'])
torch.cuda.set_device(settings['gpu'])
settings['val_batch_size']  			= 64
