import os
import shutil
import torch
import glob
import numpy as np
import pdb



def get_best_iter(exp_name):
	all_paths  							= [x for x in os.listdir(os.path.join(settings['weights_path'],exp_name)) if (('expt_data' not in x) and ('enough_iter' not in x) )]
	best_iter 							= max(set(map(lambda x : int(x.split('.')[0].split('_')[1]),all_paths)))
	return best_iter



settings   								= {}


server_root_path  						= '../../'
settings['model_dict']					= {
											'exp_name'		:'expt_resnet101_domain-net_CIPQS_R_alternate_target_source_recalc_pseudo_labels_topK_1_single_stage_nav_full_psed_dl_newaug_run1',
											'src_datasets'	:['quickdraw','painting','sketch','painting','clipart'],
											'trgt_dataset'	:'real',
										  }

settings['weights_path'] 				= os.path.join(server_root_path, 'weights')
settings['model_dict']['iterno']		= get_best_iter(settings['model_dict']['exp_name'])

dataset_name 							= 'domain-net'

settings['server_root_path']		    = server_root_path
settings['dataset_dir'] 				= os.path.join('data',dataset_name)



#dataset settings

settings['C'] 						    = ['raccoon', 'windmill', 'lightning', 'mouth', 'animal_migration', 'hammer', 'bathtub', 'television', 'birthday_cake', 'arm', 'telephone', 'firetruck', 'toothbrush', 'sailboat', 'river', 'baseball', 'scorpion', 'golf_club', 'ear', 'traffic_light', 'frying_pan', 'door', 'frog', 'keyboard', 'zebra', 'bread', 'umbrella', 'onion', 'stereo', 'dishwasher', 'pants', 'crown', 'hospital', 'square', 't-shirt', 'nail', 'squirrel', 'couch', 'hockey_puck', 'laptop', 'dog', 'steak', 'violin', 'cooler', 'bracelet', 'crab', 'cell_phone', 'shoe', 'ant', 'asparagus', 'garden_hose', 'rain', 'pineapple', 'table', 'rabbit', 'pickup_truck', 'peas', 'ice_cream', 'The_Eiffel_Tower', 'clarinet', 'string_bean', 'hedgehog', 'rollerskates', 'ceiling_fan', 'butterfly', 'flashlight', 'crayon', 'palm_tree', 'truck', 'shark', 'cat', 'binoculars', 'bee', 'star', 'foot', 'smiley_face', 'envelope', 'teddy-bear', 'feather', 'tooth', 'lion', 'toe', 'hand', 'boomerang', 'tree', 'sword', 'diamond', 'pool', 'hamburger', 'chandelier', 'floor_lamp', 'circle', 'kangaroo', 'hat', 'snake', 'house', 'sea_turtle', 'campfire', 'soccer_ball', 'megaphone', 'grass', 'jacket', 'mountain', 'cookie', 'wine_glass', 'octagon', 'church', 'cruise_ship', 'stop_sign', 'knife', 'belt', 'hurricane', 'piano', 'pear', 'wheel', 'castle', 'sink', 'headphones', 'bus', 'tennis_racquet', 'shovel', 'moon', 'hot_tub', 'trombone', 'pencil', 'duck', 'knee', 'blackberry', 'panda', 'garden', 'chair', 'tractor', 'house_plant', 'spreadsheet', 'skateboard', 'bandage', 'tornado', 'anvil', 'squiggle', 'mosquito', 'jail', 'snowflake', 'sock', 'bed', 'vase', 'baseball_bat', 'popsicle', 'bowtie', 'calculator', 'microphone', 'computer', 'wristwatch', 'shorts', 'dresser', 'mermaid', 'hexagon', 'eyeglasses', 'bulldozer', 'harp', 'fish', 'line', 'pliers', 'speedboat', 'toaster', 'banana', 'purse', 'broccoli', 'toothpaste', 'dolphin', 'bottlecap', 'washing_machine', 'bird', 'pig', 'pillow', 'parrot', 'giraffe', 'lighthouse', 'cello', 'swan', 'donut', 'lantern', 'postcard', 'eye', 'finger', 'coffee_cup', 'aircraft_carrier', 'horse', 'paint_can', 'lollipop', 'snowman', 'skyscraper', 'stitches', 'lipstick', 'camouflage', 'book', 'rake', 'hot_air_balloon', 'saxophone', 'map', 'matches', 'tiger', 'bat', 'cannon', 'hockey_stick', 'power_outlet', 'screwdriver', 'marker', 'see_saw', 'barn', 'bucket', 'ladder', 'mouse', 'underwear', 'monkey', 'leg', 'train', 'mailbox', 'basketball', 'moustache', 'microwave', 'cactus', 'rainbow', 'fireplace', 'sheep', 'tent', 'fan', 'police_car', 'sandwich', 'sleeping_bag', 'guitar', 'mug', 'face', 'compass', 'spoon', 'broom', 'alarm_clock', 'snorkel', 'nose', 'hourglass', 'drill', 'cloud', 'cake', 'skull', 'saw', 'goatee', 'cup', 'dumbbell', 'apple', 'pizza', 'canoe', 'bicycle', 'grapes', 'toilet', 'blueberry', 'submarine', 'backpack', 'stethoscope', 'airplane', 'drums', 'fence', 'teapot', 'remote_control', 'calendar', 'suitcase', 'wine_bottle', 'helicopter', 'hot_dog', 'car', 'owl', 'mushroom', 'school_bus', 'whale', 'crocodile', 'roller_coaster', 'octopus', 'basket', 'potato', 'leaf', 'zigzag', 'syringe', 'pond', 'dragon', 'triangle', 'carrot', 'parachute', 'van', 'stove', 'bridge', 'swing_set', 'bush', 'The_Great_Wall_of_China', 'helmet', 'flying_saucer', 'trumpet', 'penguin', 'candle', 'peanut', 'beard', 'fire_hydrant', 'cow', 'lighter', 'waterslide', 'elephant', 'oven', 'flower', 'spider', 'motorbike', 'strawberry', 'diving_board', 'sweater', 'paper_clip', 'scissors', 'angel', 'key', 'yoga', 'fork', 'axe', 'rhinoceros', 'brain', 'flip_flops', 'elbow', 'light_bulb', 'radio', 'camel', 'paintbrush', 'watermelon', 'The_Mona_Lisa', 'stairs', 'picture_frame', 'clock', 'sun', 'bear', 'bench', 'camera', 'eraser', 'snail', 'beach', 'ocean', 'rifle', 'passport', 'ambulance', 'lobster', 'flamingo', 'streetlight', 'necklace']
settings['num_C']						= len(settings['C'])

settings['target_dataset']				=  settings['model_dict']['trgt_dataset']


st0 = np.random.get_state()[1][0]
t0  = torch.initial_seed()

settings['seed_value']                  = {'torch':t0,'np':st0}
settings['resolution'] 					= 224

settings['index_list']					= 'index_list'

settings['bb']	 						= 'resnet101'
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

settings['gpu'] 						= 2
settings['device'] 						= 'cuda:' + str(settings['gpu'])
torch.cuda.set_device(settings['gpu'])

settings['tb_port_no'] 					= 9999-int(settings['gpu'])
settings['val_batch_size']  			= 64
settings['num_imgs']					= 4
