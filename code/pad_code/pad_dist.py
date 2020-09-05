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


MSG = 100
logging.addLevelName(MSG, "MSG")
def debugv(self, message, *args, **kws):
    if self.isEnabledFor(MSG):
        # Yes, logger takes its '*args' as 'args'.
        self._log(MSG, message, args, **kws)
logging.Logger.msg = debugv
logging.MSG = MSG



if not os.path.exists(os.path.join(config.settings['summaries_path'],config.settings['exp_name'])):
	os.makedirs(os.path.join(config.settings['summaries_path'],config.settings['exp_name'],'logval'))	
else:
	shutil.rmtree(os.path.join(config.settings['summaries_path'],config.settings['exp_name']),ignore_errors=True)
	os.makedirs(os.path.join(config.settings['summaries_path'],config.settings['exp_name'],'logval'))

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

def main():
	trainval()


def trainval():

	trainer_G = TrainerG()

	logger.msg('Loading model from iteration :{}'.format(config.settings['model_dict']['iter']))
	trainer_G.load_weights_model()
	st0 = np.random.get_state()[1][0]
	logger.msg('Tensorboard port no :{}'.format(config.settings['tb_port_no']))	
	print('seed used ={}'.format(st0))
	trainer_G.set_mode(config.settings['mode']['val'])
	trainer_G.pad_distance()
	
if __name__ == '__main__':
	main()



