import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import config as config
import numpy as np
import utils


def acc_metric(preds,labels):
	gt 	   = labels
	metric = (preds == gt).float().sum()
	return metric,gt.shape[0]


def class_wise_acc(label,pred,dom_name):
	label 		= np.array(label)
	pred 	    = np.array(pred)

	shared_trgt_catgs =  set([j for x in config.settings['src_datasets'] for j in config.settings['C'][x]])
	C_dash 		= config.settings['C_dash'][dom_name]
	catg_mappng = utils.get_catg_mapping(shared_trgt_catgs,C_dash)
	classes 	= list([ val for key,val in catg_mappng.items()])
	num_classes = len(shared_trgt_catgs) + len(config.settings['C_dash'][dom_name])
	

	avg_acc 	= {c:0 for c in classes}
	avg_count 	= {c:0 for c in classes}
	

	avg_class 	= 0

	for c in classes:
		avg_acc[c]+=np.sum((pred[label==c] == label[label==c]).astype(float))
		avg_count[c] += label[label==c].shape[0]


	for c in classes:
		if avg_count[c] == 0:
			avg_class += 0
		else:
			avg_class += (float(avg_acc[c]) / float(avg_count[c]))

	avg_class /=float(num_classes)

	return avg_class,1


def conf_acc_f_aggreement(M,gt,preds):
	K = torch.argmax(M,dim=-1)
	idx = ((K == K[:,0].unsqueeze(dim=-1)).float().prod(dim=-1) == 1)
	if (gt == preds).float().sum() >0:
		return ((gt == preds)[idx] == True).float().sum(),(gt == preds).float().sum()
	else:
		return 0,0

def conf_acc_f_dis_aggreement(M,gt,preds):
	K = torch.argmax(M,dim=-1)
	idx = ((K == K[:,0].unsqueeze(dim=-1)).float().prod(dim=-1) == 0)
	#classifiers disagree for sample
	if (gt == preds).float().sum() >0:
		return ((gt == preds)[idx] == True).float().sum(),(gt == preds).float().sum()
	else:
		return 0,0


def conf_acc_g_agreement(M,gt,preds):
	K = torch.argmax(M,dim=-1)
	idx = ((K == K[:,0].unsqueeze(dim=-1)).float().prod(dim=-1) == 1)
	return (gt[idx] == preds[idx]).float().sum(),idx[idx].shape[0]

def conf_acc_g_dis_agreement(M,gt,preds):
	K = torch.argmax(M,dim=-1)
	idx = ((K == K[:,0].unsqueeze(dim=-1)).float().prod(dim=-1) == 0)
	return (gt[idx] == preds[idx]).float().sum(),idx[idx].shape[0]



def  mcc_metric(matrix):
	b,d,c = matrix.shape
	B = b
	ls_matrix = nn.LogSoftmax(dim = -1)(matrix) #b,d,c softmax along c dimension with T temperature
	s_matrix  = nn.Softmax(dim =-1)(matrix)
	H 		  = (-1*s_matrix*ls_matrix).sum(dim=-1)    #b,d
	W 		  = B*(1 + (-1*H).exp())/(1 + (-1*H).exp()).sum(dim=0) #b,d
	W         = W.permute(1,0)
	W 		  = torch.diag_embed(W)  #d,b,b
	matrix1   = s_matrix.permute(1,2,0) #d,c,b
	matrix2   = s_matrix.permute(1,0,2) #d,b,c
	mcc  	  = torch.einsum('dcb,dbj,djk->dck',matrix1,W,matrix2) #d,c,c
	return mcc,b


def angle_cls(network):
	D = len(config.settings['src_datasets'])
	C = (config.settings['num_C'][config.settings['src_datasets'][0]])
	P_list = list(network.model['global']['M'].parameters())
	weights = P_list[0]

	weights = weights.view(D,-1,config.settings['F_dims'])

	similarity_matrix_per_class = []


	for c in range(C):
		w_dc_arr = []
		for d in range(D):
			w_dc = weights[d,c,:]
			w_dc = w_dc / torch.norm(w_dc)
			w_dc_arr.append(w_dc)

		similarity_matrix_per_class.append(torch.zeros((D, D)).cuda())

		for d1 in range(D):
			for d2 in range(D):
				w1 = w_dc_arr[d1]
				w2 = w_dc_arr[d2]
				similarity_matrix_per_class[c][d1, d2] = torch.dot(w1, w2)

	sim = torch.stack(similarity_matrix_per_class, dim=0)
	sim_per_class = sim.mean(dim=[1,2])
	avg_sim = sim_per_class.mean()
	count = 1

	return torch.acos(avg_sim)*(180/3.14),avg_sim, sim_per_class, sim, count




############################################################------------------_SECTION PLOT--------------------###########################################################################

def plot_acc_with_conf_thr(gt,preds,dom_name,conf_preds,writer,curr_iterno):
	gt 	   	   = np.array(gt)
	preds 	   = np.array(preds)
	conf_preds = np.array(conf_preds)

	idx 	   = np.argsort(conf_preds)[::-1]
	bool_list  = (gt[idx] == preds[idx]).astype(float)
	L 		   = bool_list.shape[0]
	x 		   = np.arange(1,101,1)
	y 		   = [ np.mean(bool_list[:int((i/100.)*L)]) for i in x ]

	plt.switch_backend('agg')
	fig = plt.figure()
	plt.plot(x, y)
	writer.add_figure(dom_name+'/'+"acc_conf_plot", fig, curr_iterno)
	plt.close()



def plot_acc_with_conf_thr_agreement(gt,preds,dom_name,conf_preds,all_M,writer,curr_iterno):
	gt 	   	   = np.array(gt)
	preds 	   = np.array(preds)
	conf_preds = np.array(conf_preds)

	K 		   = np.argmax(all_M,axis=-1)
	idx1 	   = (np.prod(np.float32(K == np.expand_dims(K[:,0],axis=-1)),axis=-1))== 1

	gt  	   = gt[idx1]
	preds      = preds[idx1]
	conf_preds = conf_preds[idx1]

	idx 	   = np.argsort(conf_preds)[::-1]
	bool_list  = (gt[idx] == preds[idx]).astype(float)
	L 		   = bool_list.shape[0]
	x 		   = np.arange(1,101,1)
	y 		   = [ np.mean(bool_list[:int((i/100.)*L)]) for i in x ]

	plt.switch_backend('agg')
	fig = plt.figure()
	plt.plot(x, y)
	writer.add_figure(dom_name+'/'+"acc_conf_plot_agreement", fig, curr_iterno)
	plt.close()


def plot_acc_with_conf_thr_dis_agreement(gt,preds,dom_name,conf_preds,all_M,writer,curr_iterno):
	gt 	   	   = np.array(gt)
	preds 	   = np.array(preds)
	conf_preds = np.array(conf_preds)

	K 		   = np.argmax(all_M,axis=-1)
	idx1	   = (np.prod(np.float32(K == np.expand_dims(K[:,0],axis=-1)),axis=-1))== 0

	gt  	   = gt[idx1]
	preds      = preds[idx1]
	conf_preds = conf_preds[idx1]

	idx 	   = np.argsort(conf_preds)[::-1]
	bool_list  = (gt[idx] == preds[idx]).astype(float)
	L 		   = bool_list.shape[0]
	x 		   = np.arange(1,101,1)
	y 		   = [ np.mean(bool_list[:int((i/100.)*L)]) for i in x ]

	plt.switch_backend('agg')
	fig = plt.figure()
	plt.plot(x, y)
	writer.add_figure(dom_name+'/'+"acc_conf_plot_disagreement", fig, curr_iterno)
	plt.close()
	plt.close()

############################################################------------------_SECTION LOSS--------------------###########################################################################



def l4_mirror_CE(M_logits,cls_labels):
	#by default matrix is batch x domain x class

	cls_labels = cls_labels.view(-1,1)
	n_batch,n_domain,n_class = M_logits.shape
	cls_M_logits  = M_logits.permute(0,2,1)  #batch x class x domain
	cls_labels    	   = cls_labels.expand(n_batch,n_domain)
	return nn.CrossEntropyLoss(reduction='mean')(cls_M_logits,cls_labels)


############################################################------------------_SECTION METRIC--------------------###########################################################################


def get_metric(key, feats):

	if key == 'cls_acc':
		cls_preds 		 = feats['cls_preds']
		cls_labels       = feats['cls_labels']
		return acc_metric(cls_preds,cls_labels)

	elif key == 'cls_acc_data':
		cls_preds 		 = np.array(feats['all_preds'])
		cls_labels       = np.array(feats['all_labels'])
		metric 			 = np.sum((cls_preds == cls_labels).astype(float))
		return metric,len(cls_labels)


	elif key == 'hf_acc_f_agree_recall':
		matrix  	= feats['M']
		gt 			= feats['cls_labels']
		preds       = feats['cls_preds']
		return  conf_acc_f_aggreement(matrix,gt,preds)

	elif key == 'hf_acc_f_dis_agree_recall':
		matrix  	= feats['M']
		gt 			= feats['cls_labels']
		preds       = feats['cls_preds']
		return  conf_acc_f_dis_aggreement(matrix,gt,preds)


	elif key == 'hf_acc_g_precision_agree':
		matrix  	= feats['M']
		gt 			= feats['cls_labels']
		preds       = feats['cls_preds']
		return conf_acc_g_agreement(matrix,gt,preds)

	elif key == 'hf_acc_g_precision_disagre':
		matrix  	= feats['M']
		gt 			= feats['cls_labels']
		preds       = feats['cls_preds']
		return conf_acc_g_dis_agreement(matrix,gt,preds)


	elif key == 'angle_cls':
		angle,avg, percls, sim, count = angle_cls(feats['network'])
		return angle, count


	elif key == 'cls_wise_acc':
		cls_preds 		 = feats['all_preds']
		cls_labels       = feats['all_labels']
		dom_name 		 = feats['dom_name']
		return class_wise_acc(cls_preds,cls_labels,dom_name)


	elif key == 'classifier_agreement_metric':
		M 		 = feats['M']
		K = torch.argmax(M,dim=-1)
		idx = ((K == K[:,0].unsqueeze(dim=-1)).float().prod(dim=-1) == 1)
		K = K[idx]
		return K.shape[0],float(M.shape[0])

	else:
		raise NotImplementedError('Not implemented {}'.format(key))

	return metric

############################################################------------------_SECTION LOGITS-------------------###########################################################################

def get_logits(key,feats):
	M = feats['M']/config.settings['softmax_temperature']

	if key == 8:
		cls_logits = M.softmax(dim=-1).sum(dim =1)
		return cls_logits,None,M
