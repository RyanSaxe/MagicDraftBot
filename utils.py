import torch
import numpy as np

def train(model,loss_fn,optimizer,x_data,labels,n_batches=64,epochs=20):
	"""
	function for training a model in batches. Currently no regularization
	option, but will be exploring that soon.
	"""
	losses = []
	batch_size = x_data.shape[0] // n_batches
	#create list of indices to shuffle for random batches
	idx_list = np.arange(len(x_data))
	for epoch in range(epochs):
		np.random.shuffle(idx_list)
		total_loss = 0
		accuracy = 0
		for i in range(n_batches):
			print ("\tBatch ",i,'/',n_batches)
			batch_idx = idx_list[batch_size * i:batch_size * (i+1)]
			#get batch data
			x = x_data[batch_idx,:]
			y = labels[batch_idx,:]
			#prepare model to train
			model.train()
			#make predictions
			predictions = model(x)
			#compute loss
			y_true = y.argmax(axis=1)
			loss = loss_fn(predictions,y_true)
			#compute gradients
			loss.backward()
			#update parameters
			optimizer.step()
			#compute accuracy to print out during training
			y_pred = torch.argmax(predictions,axis=1)
			amount_right = int((y_pred == y_true).sum())
			v = amount_right/len(batch_idx)
			accuracy += v
			print("\t",v)
			total_loss += loss.item()
			#zero gradients			
			optimizer.zero_grad()
		losses.append(total_loss)
		print('Epoch',epoch,': Accuracy = ',accuracy/n_batches," Total Loss = ",total_loss)
	return losses

def generate_pack(card_df):
	"""
	generate random pack of MTG cards
	"""
	p_r = 7/8
	p_m = 1/8
	if random.random() < 1/8:
		rare = random.sample(card_df[card_df['rarity'] == 'mythic'].index.tolist(),1)
	else:
		rare = random.sample(card_df[card_df['rarity'] == 'rare'].index.tolist(),1)
	uncommons = random.sample(card_df[card_df['rarity'] == 'uncommon'].index.tolist(),3)
	commons = random.sample(card_df[card_df['rarity'] == 'common'].index.tolist(),10)
	idxs = rare + uncommons + commons
	pack = torch.zeros(len(card_df))
	pack[idxs] = 1
	return pack

def read_log(fname,card_df);
	"""
	process MTGO log file and convert it into tensors so the bot
	can say what it would do
	"""
	with open(fname,'r') as f:
		lines = f.readlines()
	set_lookup = {v:i for i,v in enumerate(card_df['name'].to_dict())}
	packs = []
	picks = []
	pools = []
	in_pack = False
	cur_pack = np.zeros(len(set_lookup.keys()))
	cur_pick = np.zeros(len(set_lookup.keys()))
	pool = np.zeros(len(set_lookup.keys()))
	for line in lines:
		match = re.findall(r'Pack \d pick \d+',line)
		if len(match) == 1:
			in_pack = True
			continue
		if in_pack:
			if len(line.strip()) == 0:
				in_pack = False
				if sum(cur_pick) != 0:
					packs.append(cur_pack)
					picks.append(cur_pick)
					pools.append(pool.copy())
					pool += cur_pick
				cur_pack = np.zeros(len(set_lookup.keys()))
				cur_pick = np.zeros(len(set_lookup.keys()))
				continue
			process = line.strip()
			if process.startswith("-"):
				cardname = process.split(' ',1)[1].replace(' ','_').replace(',','')
				if cardname in ignore_cards:
					continue
				card_idx = set_lookup[cardname]
				cur_pick[card_idx] = 1
			else:
				cardname = process.replace(' ','_').replace(',','')
				if cardname in ignore_cards:
					continue
				card_idx = set_lookup[cardname]
			cur_pack[card_idx] = 1
	return pools,picks,packs