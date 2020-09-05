import torch
import torch.nn as nn
from torchvision import models
from collections import OrderedDict


class MatrixLayer(nn.Module):

	def __init__(self, input_dim, domains, classes):

		super(MatrixLayer, self).__init__()
		self.domains = domains
		self.classes = classes
		self.input_dim = input_dim
		self.linear = nn.Linear(self.input_dim, self.domains * self.classes)

	def forward(self, x):
		l = self.linear(x)
		return l.view((-1, self.domains, self.classes))


class IdentityLayer(nn.Module):

	def __init__(self):
		super(IdentityLayer, self).__init__()

	def forward(self, x):
		return x


class ForwardLayer(nn.Module):

	def __init__(self, inp_lin1, inp_lin2, f_dims):
		super(ForwardLayer, self).__init__()
		self.inp_lin1 = inp_lin1
		self.inp_lin2 = inp_lin2
		self.f_dims = f_dims

		self.net = nn.Sequential(
				nn.Linear(self.inp_lin1,self.inp_lin2),
				nn.ELU(),
				nn.Linear(self.inp_lin2,self.inp_lin2),
				nn.BatchNorm1d(self.inp_lin2),
				nn.ELU(),
				nn.Linear(self.inp_lin2,self.f_dims),
				nn.ELU(),
				nn.Linear(self.f_dims, self.f_dims),
				nn.BatchNorm1d(self.f_dims),
				nn.ELU()
			)

		
	def forward(self,x,return_all_layers=False):
		features_list = []
		f = x
		for l in self.net.children():
			f = l(f)
			features_list.append(f)

		if return_all_layers:
			return features_list
		else:
			return features_list[-1]


class BackBoneLayer(nn.Module):

	def __init__(self,pre,out_feats):
		super(BackBoneLayer, self).__init__()

		if pre == 'resnet101':
			temp_resnet = models.resnet101(pretrained=True)
			self.features = nn.Sequential(*[x for x in list(temp_resnet.children())[:-1]])
		elif pre == 'resnet50':
			temp_resnet = models.resnet50(pretrained=True)
			self.features = nn.Sequential(*[x for x in list(temp_resnet.children())[:-1]])
	
		self.out_feats = out_feats

		self.layer_names = [ name for name,a in temp_resnet.named_children()]
		self.layers_outs = []

	def get_layers_output(self):
		layers_dict = {name:out for name,out in zip(self.layer_names,self.layers_outs)}
		return layers_dict


	def forward(self, x):
		feats = self.features(x)		
		return feats.view((x.shape[0], self.out_feats))




