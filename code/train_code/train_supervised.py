from time import time
from tensorboardX import SummaryWriter
import config as config
from tabulate import tabulate
from trainer_supervised import TrainerG
import shutil
import os
import warnings
import torch
import numpy as np
import logging
import sys
import dataset


#code for logging during training
MSG = 100
logging.addLevelName(MSG, "MSG")
def debugv(self, message, *args, **kws):
    if self.isEnabledFor(MSG):
        self._log(MSG, message, args, **kws)
logging.Logger.msg = debugv
logging.MSG = MSG

if not os.path.exists(os.path.join(config.settings['summaries_path'],config.settings['exp_name'])):
		os.mkdir(os.path.join(config.settings['summaries_path'],config.settings['exp_name']))
else:
	if config.settings['load_model'] ==False and config.settings['load_opt'] == False and config.settings['start_iter'] == 1:
			shutil.rmtree(os.path.join(config.settings['summaries_path'],config.settings['exp_name']),ignore_errors=True)
			os.mkdir(os.path.join(config.settings['summaries_path'],config.settings['exp_name']))



if config.settings['load_model'] ==False and config.settings['load_opt'] == False and config.settings['start_iter'] == 1:
	logger = logging.getLogger()
	logger.setLevel(logging.MSG)
	fh = logging.FileHandler(os.path.join(config.settings['summaries_path'], config.settings['exp_name'],'log.txt'),mode='w')
	fh.setLevel(logging.MSG)
	ch = logging.StreamHandler(sys.stdout)
	ch.setLevel(logging.MSG)
	formatter = logging.Formatter('%(message)s')
	fh.setFormatter(formatter)
	ch.setFormatter(formatter)
	logger.addHandler(fh)
	logger.addHandler(ch)
else:
	logger = logging.getLogger()
	logger.setLevel(logging.MSG)
	fh = logging.FileHandler(os.path.join(config.settings['summaries_path'], config.settings['exp_name'],'log.txt'),mode='a')
	fh.setLevel(logging.MSG)
	ch = logging.StreamHandler(sys.stdout)
	ch.setLevel(logging.MSG)
	formatter = logging.Formatter('%(message)s')
	fh.setFormatter(formatter)
	ch.setFormatter(formatter)
	logger.addHandler(fh)
	logger.addHandler(ch)





logger = logging.getLogger()
warnings.simplefilter("ignore", UserWarning)

def str_gen(item):
	a = ''
	if isinstance(item,dict):
		for key,value in item.items():
			a+=str(key)+':'+str(value)+'\n'
		return a
	else:
		return str(item)

table =[]
for key,value in config.settings.items():
	ke_val = [key,str_gen(value)]
	table.append(ke_val)


if not config.settings['quite_mode']:
	logger.msg('--------- SANITY CHECK ---------')
	logger.msg(tabulate(table,headers=['No','Settings','Value'],tablefmt="simple",showindex=True))

	ip = raw_input('continue? (y/n): ')
	ip = str(ip)

	if ip.lower() == 'y' or str(ip.lower()) == 'yes':
		pass
	else:
		logger.msg('Decided not to execute!')
		exit()


def main():

	if not os.path.exists(os.path.join(config.settings['weights_path'],config.settings['exp_name'])):
		os.mkdir(os.path.join(config.settings['weights_path'],config.settings['exp_name']))
	else:
		if config.settings['load_model'] ==False and config.settings['load_opt'] == False and config.settings['start_iter'] == 1:
			shutil.rmtree(os.path.join(config.settings['weights_path'],config.settings['exp_name']))
			os.mkdir(os.path.join(config.settings['weights_path'],config.settings['exp_name']))

	if not os.path.exists(os.path.join(config.settings['summaries_path'],config.settings['exp_name'],'logdir_train')):
		os.mkdir(os.path.join(config.settings['summaries_path'],config.settings['exp_name'],'logdir_train'))
		os.mkdir(os.path.join(config.settings['summaries_path'],config.settings['exp_name'],'logdir_val'))
	else:
		if config.settings['load_model'] ==False and config.settings['load_opt'] == False and config.settings['start_iter'] == 1:
			shutil.rmtree(os.path.join(config.settings['summaries_path'],config.settings['exp_name'],'logdir_train'))
			shutil.rmtree(os.path.join(config.settings['summaries_path'],config.settings['exp_name'],'logdir_val'))
			os.mkdir(os.path.join(config.settings['summaries_path'],config.settings['exp_name'],'logdir_train'))
			os.mkdir(os.path.join(config.settings['summaries_path'],config.settings['exp_name'],'logdir_val'))

	if config.settings['load_model'] ==False and config.settings['load_opt'] == False and config.settings['start_iter'] == 1:
		with open(os.path.join(config.settings['summaries_path'], config.settings['exp_name'], 'config.txt'), 'w') as history_file:
			logger.msg('saving in ' + os.path.join(config.settings['summaries_path'], config.settings['exp_name'], 'config.txt'))
			history_file.write('\n===== x ===== x =====\n')
			history_file.write(tabulate(table,headers=['No','Settings','Value'],tablefmt="simple",showindex=True))
	else:
		with open(os.path.join(config.settings['summaries_path'], config.settings['exp_name'], 'config.txt'), 'a') as history_file:
			logger.msg('saving in ' + os.path.join(config.settings['summaries_path'], config.settings['exp_name'], 'config.txt'))
			history_file.write('\n===== x ===== x =====\n')
			history_file.write(tabulate(table,headers=['No','Settings','Value'],tablefmt="simple",showindex=True))

	trainval()





def trainval():

	trainer_G = TrainerG()

	#dataset.load_dataset()
	
	if config.settings['load_model']:
		logger.msg('Resuming training from iteration :{}'.format(config.settings['start_iter']))
		trainer_G.load_weights_model()

	if config.settings['load_opt']:
		trainer_G.load_optimizers()

	if config.settings['continue_training']:
		trainer_G.initialize_pseudo_target_indices()
		for dom in config.settings['trgt_datasets']:
			trainer_G.initialize_pseudo_trgt_dataloader(dom)

	if not config.settings['quite_mode']:
		config.start_tb()

	enough_iter     = config.settings['enough_iter']
	max_iter     	= config.settings['max_iter']
	val_after 		= config.settings['val_after']
	batch_size 		= config.settings['batch_size']
	val_batch_size  = int(config.settings['val_batch_size_factor']*config.settings['batch_size'])

	logger.msg('enough_iter:{}\nmax_iter:{}\nval_after:{}\ntraining batch_size for each domain :{}\nval_batch_size:{}\n'.format(enough_iter,max_iter,val_after,batch_size,val_batch_size))
	
	while True:

		st0 = np.random.get_state()[1][0]

		if trainer_G.current_iteration % config.settings['log_interval'] == 0 or trainer_G.current_iteration % config.settings['log_interval'] == 1:
			logger.msg ("\n----------- train_iter " + str(trainer_G.current_iteration) + ' -----------\n')
			logger.msg('From train: torch_seed={} numpy seed={}'.format(torch.initial_seed(),st0))
			print('Running dataset = {}, expt = {}'.format(config.dataset_name,config.data_key))

		if not config.settings['quite_mode']:
			if trainer_G.current_iteration % config.settings['log_interval'] == 0 or trainer_G.current_iteration % config.settings['log_interval'] == 1:
				logger.msg('Tensorboard port no :{}'.format(config.settings['tb_port_no']))

		trainer_G.set_mode(config.settings['mode']['train'])
		trainer_G.train()
		
		if trainer_G.current_iteration%val_after == 0:
			if not config.settings['validate_target_before_enough_iters']:
				if trainer_G.current_iteration >= max(config.settings['val_after'],config.settings['enough_iter']):
					logger.msg('validating')
					trainer_G.set_mode(config.settings['mode']['val'])
					trainer_G.val_over_target_set()
			else:
				logger.msg('validating')
				trainer_G.set_mode(config.settings['mode']['val'])
				trainer_G.val_over_target_set()

		

		if trainer_G.current_iteration > config.settings['max_iter']:
			break

		trainer_G.current_iteration+=1



if __name__ == '__main__':
	main()



