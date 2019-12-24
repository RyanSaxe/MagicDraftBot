from sklearn.cluster import KMeans
import torch
import random

def add_clusters(features,drafts,n_archetypes,colors_only=False,save=True):
	"""
	kmeans clustering of the draft data
	"""
	draft_pool_vectors = aggregate_drafts(drafts[:,:,1:],features,colors_only=colors_only)
	kmeans = KMeans(n_clusters=n_archetypes).fit(draft_pool_vectors)
	if save:
		torch.save(kmeans,'Saved_Models/clusters.pkl')
	return kmeans
def aggregate_drafts(drafts,features,colors_only=False):
	"""
	logic for aggregating all picks of a draft into one vector
	so that the clustering algorithm can consider a draft as a 
	single data point
	"""
	n_cards = features.index.size
	#this grabs the pool during the last pick
	last_picks = drafts[:,-1,:n_cards]
	#binary feature to describe the colors of a card
	colors = features[list('WUBRG')]
	#compute the color density of a draft pool
	color_density = torch.matmul(last_picks,torch.tensor(colors.values).type(torch.float))
	#return the draft pool with 6 additional features to describe coor density
	if colors_only:
		return color_density
	else:
		return torch.cat([last_picks,color_density],1)

def create_drafts():
	"""
	this is the function for converting the main DraftSim data into one-hot-encoded vectors.

	The actual code and pickled files are excluded from the repo due to NDA, but
	this is the body replacing it in order to test that the pipeline works.
	"""
	draft_picks = torch.load('Data/dpicks.pkl')
	draft_packs = torch.load('Data/dpacks.pkl')
	return draft_packs,draft_picks

def get_format_features():
	"""
	this is the function for grabbing card features such as color and name.

	#the code to create ft.pkl is replaced by loading it due to NDA
	"""
	return torch.load('Data/ft_full.pkl')

def create_dataset(n_archetypes=15,full_dataset=False,save_clusters=True):
	#get colors of each card in the set
	features = get_format_features()
	#create embedding from card_name to integer
	card_mapping = features['name'].to_dict()
	#get gets the draft data and converts to the following via one-hot-encoding:
	#	A X B X C matrix = Draft Packs
	#		A is number of drafts in the dataset
	#		B is number of picks in the draft (42)
	#		C is 2 * (number of cards in the set) + 1 (499)
	#			on the C axis, the first element is empty and 
	#			will be populated via clustering. The next 249
	#			is the draft pool. And the next 249 is the current
	#			pack. 
	#	A X 249 = Draft Picks --> binary vector for the correct pick
	draft_packs,draft_picks = create_drafts()
	#cluster the dataset via archetype
	clusters = add_clusters(features,draft_packs,n_archetypes,save=save_clusters)
	#update the data to include the cluster
	draft_packs[:,:,0] = torch.tensor(clusters.labels_)[:,None]
	#note, currently I do not include the extra features into the
	#data, but that is one of the next steps I intend to take.
	if full_dataset:
		train_perc = 1
	else:
		train_perc = 0.8
	#very important to divide train,test by full draft and not
	#individual picks to avoid leakage.
	size = draft_packs.shape[0]
	train_size = int(size * train_perc)
	train_idx = random.sample(range(size),train_size)
	test_idx = list(set(range(size)) - set(train_idx))
	train_pack = draft_packs[train_idx,:,:]
	train_pick = draft_picks[train_idx,:,:]
	test_pack = draft_packs[test_idx,:,:]
	test_pick = draft_picks[test_idx,:,:]
	#note: no need for validation set since there is no hyperparameter
	#tuning or feedback given by it at the moment. This is another aspect
	#to add in the future
	return train_pack,train_pick,test_pack,test_pick

