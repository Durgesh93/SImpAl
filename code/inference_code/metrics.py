import config

def get_logits(key,feats):
	M = feats['M']/config.settings['softmax_temperature']
	if key == 8:
		cls_logits = M.softmax(dim=-1).sum(dim =1)
		return cls_logits,None,M