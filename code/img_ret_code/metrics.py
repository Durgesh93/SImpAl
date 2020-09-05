import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import config as config
import numpy as np
import utils
import pdb


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

def entropy_smax(logits):
	S = nn.Softmax(dim = -1)
	LS = nn.LogSoftmax(dim = -1)
	b = -1*S(logits) * LS(logits)
	b = b.sum()
	return b,logits.shape[0]

def norm_entropy(logits):
	logits= logits/torch.sum(logits,dim=-1,keepdim=True)
	LS = nn.LogSoftmax(dim = -1)
	b = -1*logits * LS(logits)
	b = b.sum()
	return b,logits.shape[0]

def global_entropy(logits):
	smax_logits = F.softmax(logits,dim=-1)
	b = -1* smax_logits*smax_logits.log()
	return b.sum(),b.shape[0]



def disc_norm(M,p=None):
	n_batch,n_domain,n_class = M.shape
	loss = 0
	for i in range(0,n_domain):
		for j in range(i+1,n_domain):
			c_i = M[:,i,:]
			c_j = M[:,j,:]
			loss+=torch.sum(torch.norm(c_i-c_j,dim=-1,p=p)**2,dim=0)
	return loss,M.shape[0]

def disc_norm_wp(M,p=None):
	n_batch,n_domain,n_class = M.shape
	_,_,M = get_logits(key=config.settings['target_label_logit_key'],feats={'M':M})

	loss = 0
	for i in range(0,n_domain):
		for j in range(i+1,n_domain):
			c_i = M[:,i,:]
			c_j = M[:,j,:]
			loss+=torch.sum(torch.norm(c_i-c_j,dim=-1,p=p)**2,dim=0)
	return loss,M.shape[0]



def disc_kl(M):
	n_batch,n_domain,n_class = M.shape
	loss = 0
	for i in range(0,n_domain):
		for j in range(0,n_domain):
			if i!=j:
				c_i = M[:,i,:]
				c_j = M[:,j,:]
				s_ci = F.softmax(c_i, dim=-1)
				s_cj = F.softmax(c_j, dim=-1)
				loss+= (s_ci * (s_ci / s_cj).log()).sum()
	return loss,M.shape[0]

def disc_cent(M):
	n_batch,n_domain,n_class = M.shape
	loss = 0
	for i in range(0,n_domain):
		for j in range(0,n_domain):
			if i!=j:
				c_i = M[:,i,:]
				c_j = M[:,j,:]
				s_ci = F.softmax(c_i, dim=-1)
				s_cj = F.softmax(c_j, dim=-1)
				loss+= (-s_ci * s_cj.log()).sum()
	return loss,M.shape[0]


def high_conf_acc_f(M,gt,preds):
	K = torch.argmax(M,dim=-1)
	idx = ((K == K[:,0].unsqueeze(dim=-1)).float().prod(dim=-1) == 1)
	if (gt == preds).float().sum() >0:
		return ((gt == preds)[idx] == True).float().sum(),(gt == preds).float().sum()
	else:
		return 0,1


def high_conf_acc_g(M,gt,preds):
	K = torch.argmax(M,dim=-1)
	idx = ((K == K[:,0].unsqueeze(dim=-1)).float().prod(dim=-1) == 1)
	return (gt[idx] == preds[idx]).float().sum(),idx.shape[0]



def avg_max_conf(M):
	M = M.softmax(dim=-1)
	return torch.max(M,dim=-1)[0].sum(),M.shape[0]

def avg_logit_diff(M):
	M = M.softmax(dim=-1)
	sorted_logits = torch.sort(M,dim=-1,descending=True)[0]
	return (sorted_logits[:,:,0] - sorted_logits[:,:,1]).sum(),sorted_logits.shape[0]


def avg_max_conf_hf_g(M):
	M = M.softmax(dim=-1)
	K = torch.argmax(M,dim=-1)
	idx = ((K == K[:,0].unsqueeze(dim=-1)).float().prod(dim=-1) == 1)
	sorted_logits = torch.sort(M,dim=-1,descending=True)[0]
	return (sorted_logits[:,:,0])[idx].sum(),idx.shape[0]


def avg_logit_diff_hf_g(M):
	M = M.softmax(dim=-1)
	K = torch.argmax(M,dim=-1)
	idx = ((K == K[:,0].unsqueeze(dim=-1)).float().prod(dim=-1) == 1)
	sorted_logits = torch.sort(M,dim=-1,descending=True)[0]
	return (sorted_logits[:,:,0] - sorted_logits[:,:,1])[idx].sum(),idx.shape[0]

	

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

############################################################------------------_SECTION LOSS--------------------###########################################################################

def l1_class_CE(M_logits,cls_labels):
	dom_sp_ce_loss = 0
	tot_batch_size = M_logits.shape[0]

	cls_labels = cls_labels.view(len(config.settings['src_datasets']),-1)
	M_logits = M_logits.view(len(config.settings['src_datasets']),config.settings['batch_size'],len(config.settings['src_datasets']),-1)

	for d in range(M_logits.shape[0]):
	 	M_logits_d = M_logits[d,:,d,:]
	 	cls_labels_d 	= cls_labels[d]
	 	dom_sp_ce_loss += F.cross_entropy(M_logits_d, cls_labels_d)

	dom_sp_ce_loss /= tot_batch_size

	return dom_sp_ce_loss

def l4_mirror_weighted_CE(M_logits,cls_labels,weight):
	#by default matrix is batch x domain x class
	
	n_batch,n_domain,n_class = M_logits.shape
	cls_labels  			 = cls_labels.view(-1,1)
	cls_M_logits  			 = M_logits.permute(0,2,1)  #batch x class x domain

	cls_labels_one_hot 		 = torch.FloatTensor(n_batch,n_class).to(config.settings['device'])
	cls_labels_one_hot.zero_()
	cls_labels_one_hot.scatter_(1, cls_labels, 1)
	cls_labels_one_hot 		 = cls_labels_one_hot.view(n_batch,n_class,1).expand(n_batch,n_class,n_domain) #batch x class x domain
	weight  				 = weight.view(n_batch,1,1)                                                    # batch,1,1
	loss 					 = - F.log_softmax(cls_M_logits,dim=1)*cls_labels_one_hot*weight
	return 	loss.mean()

	
def l2_domain_CE1(M_logits,dom_labels,cls_labels):
    # For GT class
    loss = 0
    for i,c in enumerate(cls_labels):
        L = M_logits[i,:,c] # D
        s = F.softmax(L, dim=-1)
        ce = - s * torch.log(s) # D
        d = dom_labels[i]
        loss += ce[d]
    loss /= cls_labels.shape[0]
    return loss


def l2_domain_CE2(M_logits,dom_labels,cls_labels):
    # Everything but the GT class
    loss = 0
    for i,c in enumerate(cls_labels):
        L = M_logits[i,:,:] # D x C
        s = F.softmax(L, dim=0)
        ce = - s * torch.log(s) # D x C
        d = dom_labels[i]
        ce_all_classes = ce[d] # C
        ce_all_except_c = ce_all_classes.sum() - ce_all_classes[c]
        loss += ce_all_except_c
    loss /= cls_labels.shape[0]
    return loss

def l2_domain_CE3(M_logits,dom_labels):
    b,d,c = M_logits.shape
    dom_M_logits = M_logits
    return F.cross_entropy(dom_M_logits,dom_labels.unsqueeze(dim=1).repeat(1,c))

def l4_mirror_CE(M_logits,cls_labels):
	#by default matrix is batch x domain x class

	cls_labels = cls_labels.view(-1,1)
	n_batch,n_domain,n_class = M_logits.shape
	cls_M_logits  = M_logits.permute(0,2,1)  #batch x class x domain
	cls_labels    	   = cls_labels.expand(n_batch,n_domain)
	return nn.CrossEntropyLoss(reduction='mean')(cls_M_logits,cls_labels)


def l3_global_CE(M_logits,cls_labels):
	cls_logits,_,_ = get_logits(key=settings['target_label_logit_key'],feats={'matrix':M_logits})
	return F.cross_entropy(cls_logits, cls_labels)


def l3_global_NLL(M_logits,cls_labels):
	cls_logits,_,_ = get_logits(key=settings['target_label_logit_key'],feats={'matrix':M_logits})
	return F.nll_loss(cls_logits, cls_labels)


def l5_cls_and_dom_CE(M_logits,cls_labels,dom_labels):
	M_logits = M_logits.view(M_logits.shape[0],-1)
	n_C = config.settings['num_C'][config.settings['src_datasets'][0]]+config.settings['num_C_dash'][config.settings['src_datasets'][0]]
	labels        = dom_labels*n_C+cls_labels
	return F.cross_entropy(M_logits,labels)



############################################################------------------_SECTION METRIC--------------------###########################################################################


def get_metric(key, feats):

	if key == 'cls_acc' or key =='dom_acc':
		cls_preds 		 = feats['cls_preds']
		cls_labels       = feats['cls_labels']
		return acc_metric(cls_preds,cls_labels)

	elif key == 'entropy_norm':
		cls_logits 		 = feats['cls_logits']
		return norm_entropy(cls_logits)

	elif key == 'entropy_smax':
		cls_logits 		 = feats['cls_logits']
		return entropy_smax(cls_logits)

	elif key == 'entropy_global':
		cls_logits 		 = feats['cls_logits']
		return global_entropy(cls_logits)


	elif key == 'disc_norm_wp':
		M      = feats['M']
		pord   = feats['p']
		return disc_norm_wp(M,pord)

	elif key == 'disc_norm_c':
		M 		= feats['M']
		pord    = feats['p']
		return disc_norm(M,pord)

	elif key == 'disc_norm_smax':
		M 		= feats['M']
		pord    = feats['p']
		return disc_norm(F.softmax(M,dim=-1),pord)

	elif key == 'disc_kl':
		M 		= feats['M']
		return disc_kl(M)

	elif key == 'disc_cent':
		M 		= feats['M']
		return disc_cent(M)

	elif key == 'avg_max_conf':
		M 		= feats['M']
		return avg_max_conf(M)

	elif key == 'avg_logit_diff':
		M 		= feats['M']
		return avg_logit_diff(M)

	elif key == 'avg_max_conf_hf_g':
		M 		= feats['M']
		return avg_max_conf_hf_g(M)

	elif key == 'avg_logit_diff_hf_g':
		M 		= feats['M']
		return avg_logit_diff_hf_g(M)


	elif key == 'mcc':
		matrix  	= feats['matrix']
		return mcc_metric(matrix)

	elif key == 'hf_acc_f_recall':
		matrix  	= feats['M']
		gt 			= feats['cls_labels']
		preds       = feats['cls_preds']
		return  high_conf_acc_f(matrix,gt,preds)

	elif key == 'hf_acc_g_precision':
		matrix  	= feats['M']
		gt 			= feats['cls_labels']
		preds       = feats['cls_preds']
		return high_conf_acc_g(matrix,gt,preds)


	elif key == 'angle_cls':
		angle,avg, percls, sim, count = angle_cls(feats['network'])
		return angle, count


	elif key == 'cls_wise_acc':
		cls_preds 		 = feats['all_preds']
		cls_labels       = feats['all_labels']
		dom_name 		 = feats['dom_name']
		return class_wise_acc(cls_preds,cls_labels,dom_name)

	elif key == 'cls_acc_data':
		cls_preds 		 = np.array(feats['all_preds'])
		cls_labels       = np.array(feats['all_labels'])
		metric 			 = np.sum((cls_preds == cls_labels).astype(float))
		return metric,len(cls_labels)

	elif key == 'cls_batch_acc':
		cls_preds 		 = feats['cls_labels']
		cls_labels       = feats['cls_preds']
		metric 			 = (cls_preds == cls_labels).float().sum()
		return metric/len(cls_labels),1


	elif key == 'classifier_agreement_metric':
		M 		 = feats['M']
		K = torch.argmax(M,dim=-1)
		idx = ((K == K[:,0].unsqueeze(dim=-1)).float().prod(dim=-1) == 1)
		K = K[idx]
		return K.shape[0],float(M.shape[0])

	elif key == 'mirror_loss_target_train':
		M 		 = feats['M']
		K = torch.argmax(M,dim=-1)
		idx = ((K == K[:,0].unsqueeze(dim=-1)).float().prod(dim=-1) == 1)
		K = K[idx]
		labels = K[:,0]
		if len(labels) == 0:
			return 0,1
		else:
			return l4_mirror_CE(M[idx],labels),1

	else:
		raise NotImplementedError('Not implemented {}'.format(key))

	return metric


############################################################------------------_SECTION LOGITS-------------------###########################################################################

def get_logits(key,feats):
	M = feats['M']/config.settings['softmax_temperature']

	W = F.softmax(M,dim=1)
	P = F.softmax(M,dim=-1)

	max_P,_ = P.max(dim=-1,keepdim=True)
	max_W,_ = W.max(dim=1,keepdim=True)

	if key == 1:
		cls_logits = M.sum(dim =1).softmax(dim=-1)
		dom_logits = M.sum(dim =-1).softmax(dim=1)
		return cls_logits,dom_logits,M

	elif key == 2:

		W_M = W*M
		P_M = P*M
		cls_logits = F.softmax(W_M.sum(dim=1),dim=-1)
		dom_logits = F.softmax(P_M.sum(dim=-1),dim=1)

		return cls_logits,dom_logits,W_M

	elif key == 3:

		mW_M = max_W*M
		mP_M = max_P*M

		cls_logits = F.softmax(mW_M.sum(dim=1),dim=-1)
		dom_logits = F.softmax(mP_M.sum(dim=-1),dim=1)

		return cls_logits,dom_logits,mW_M

	elif key == 4:

		mW_W = max_W*W
		mP_P = max_P*P
		mW_W = mW_W.sum(dim=1) #b,c
		mP_P = mP_P.sum(dim=-1) #b,d
		cls_logits = mW_W/mW_W.sum(dim=-1).unsqueeze(dim=-1)
		dom_logits = mP_P/mP_P.sum(dim=-1).unsqueeze(dim=-1)
		return cls_logits,dom_logits,mW_W

	elif key == 5:

		W_P = W*P
		logits 	   = W_P.sum(dim=1)
		cls_logits = logits/torch.sum(logits,dim=-1).unsqueeze(dim=-1)
		d_logits  = W_P.sum(dim=-1)
		dom_logits  = d_logits/torch.sum(d_logits,dim=-1).unsqueeze(dim=-1)
		return cls_logits,dom_logits,W_P

	elif key == 6:
		cls_logits = P.sum(dim=1)
		dom_logits = W.sum(dim=-1)
		cls_logits = cls_logits/cls_logits.sum(dim=-1,keepdim=True)
		dom_logits = dom_logits/dom_logits.sum(dim=-1,keepdim=True)
		return cls_logits,dom_logits,P

	elif key == 7:
		cls_logits = M.sum(dim =1)
		dom_logits = M.sum(dim =-1)
		return cls_logits,dom_logits,M

	elif key == 8:
		cls_logits = M.softmax(dim=-1).sum(dim =1)
		return cls_logits,None,M

	elif key == 9:
		
		M_sorted,_   = torch.sort(M,dim=-1,descending=True)
		weights_cls  = (M_sorted[:,:,0]-M_sorted[:,:,1]).unsqueeze(dim=-1)
		M_cls        = P*weights_cls
		cls_logits   = M_cls.sum(dim=1)
		return cls_logits,None,M_cls

	elif key == 10:
		
		P_sorted,_   = torch.sort(P,dim=-1,descending=True)
		weights_cls  = (P_sorted[:,:,0]-P_sorted[:,:,1]).unsqueeze(dim=-1)
		M_cls        = M*weights_cls

		W_sorted,_   = torch.sort(W,dim=1,descending=True)
		weights_dom  = (W_sorted[:,0,:]-W_sorted[:,1,:]).unsqueeze(dim=1)
		M_dom        = M*weights_dom

		cls_logits   = M_cls.sum(dim=1)
		dom_logits   = M_dom.sum(dim=-1)

		return cls_logits,dom_logits,M_cls
