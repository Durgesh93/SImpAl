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



print('--------- SANITY CHECK ---------')
print(tabulate(table,headers=['No','Settings','Value'],tablefmt="simple",showindex=True))

ip = raw_input('continue? (y/n): ')
ip = str(ip)

if ip.lower() == 'y' or str(ip.lower()) == 'yes':
	pass
else:
	print('Decided not to execute!')
	exit()


def main():
	trainval()





def trainval():
	trainer_G = TrainerG()
	trainer_G.load_weights_model()
	
	if config.dataset_name == 'domain-net':
		trainer_G.prepare_indices_domain_net()
	else:
		trainer_G.prepare_indices_generic()

	trainer_G.set_mode(config.settings['mode']['val'])
	trainer_G.val_over_target_set()


if __name__ == '__main__':
	main()



