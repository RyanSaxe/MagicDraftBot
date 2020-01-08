
import sys
import torch
from models import *
from preprocessing import *
import utils
import matplotlib.pyplot as plt
#script for creating new models with the same initial specifications

def create_dataset_test():
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
	clusters = torch.load('Saved_Models/clusters_final.pkl')
	#update the data to include the cluster
	draft_packs[:,:,0] = torch.tensor(clusters.labels_)[:,None]
	#note, currently I do not include the extra features into the
	#data, but that is one of the next steps I intend to take.
	#if full_dataset:
	#	train_perc = 1
	#else:
	#	train_perc = 0.8
	#very important to divide train,test by full draft and not
	#individual picks to avoid leakage.
	#size = draft_packs.shape[0]
	#train_size = int(size * train_perc)
	train_idx = torch.load('Data/train_idx_final.pkl')
	test_idx = torch.load('Data/test_idx_final.pkl')
	#if not full_dataset:
	#	#store the train/test split for the current model
	#	torch.save(test_idx,'Data/test_idx_new.pkl')
		#torch.save(train_idx,'Data/train_idx_new.pkl')
	train_pack = draft_packs[train_idx,:,:]
	train_pick = draft_picks[train_idx,:,:]
	test_pack = draft_packs[test_idx,:,:]
	test_pick = draft_picks[test_idx,:,:]
	#note: no need for validation set since there is no hyperparameter
	#tuning or feedback given by it at the moment. This is another aspect
	#to add in the future
	return train_pack,train_pick,test_pack,test_pick

ending = "_" + sys.argv[1]
loss_function = torch.nn.CrossEntropyLoss()
train_packs,train_picks,test_packs,test_picks = create_dataset_test()
rank_model = torch.load('Saved_Models/rank_model_final.pkl')
#initialize drafting model with learned weights from rank model
init_weights = rank_model.rank_matrix.clone().detach()
#normalize the weights such that 1 is the largest initial weight
smaller_init_weights = init_weights / init_weights.max(0, keepdim=True)[0]
draft_model = DraftNet(smaller_init_weights)
#add l2 regularization to avoid exploding weights
#with regularization, also lower the learning rate and increase epochs
#note: with this regularization there is no need for ceiling on pool bias.
optimizer = torch.optim.Adam(draft_model.parameters(), lr=0.01,weight_decay=1e-5)
#flatten the drafts so that the algorithm only considers each pick
#individually and remove archetype label to avoid leakage
train_x = torch.flatten(train_packs,start_dim=0,end_dim=1)[:,1:]
train_y = torch.flatten(train_picks,start_dim=0,end_dim=1)
#train the model
losses = utils.train(draft_model,loss_function,optimizer,train_x,train_y,epochs=100)
#flatten test data and remove archetype label to avoid leakage
test_x = torch.flatten(test_packs,start_dim=0,end_dim=1)[:,1:]
test_y = torch.flatten(test_picks,start_dim=0,end_dim=1)
#make predictions
npicks = 42
accuracy = []
for pick in range(npicks):
	idx = list(range(pick,test_x.shape[0],npicks))
	x = test_x[idx]
	y = test_y[idx]
	predictions = draft_model(x)
	y_pred = torch.argmax(predictions,axis=1)
	y_true = torch.argmax(y,axis=1)
	#evaluate predictions
	amount_right = int((y_pred == y_true).sum())
	acc = amount_right/y.shape[0]
	print("Pick",pick + 1,"Accuracy: ",acc)
	accuracy.append(acc)
plt.plot(accuracy)
plt.xlabel('Time (pick number)')
plt.ylabel('Accuracy')
plt.title('Avg Accuracy in Test Set')
plt.savefig('Output/accuracy_curve' + ending + '.png')
#save the model so it can be used to make decisions in a real draft
torch.save(losses,'Data/train_loss' + ending + '.pkl')
torch.save(draft_model,'Saved_Models/draft_model' + ending + '.pkl')