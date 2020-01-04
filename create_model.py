from preprocessing import create_dataset
from models import *
import torch
import matplotlib.pyplot as plt
import utils
import sys

#flags for train-test-split and saving the models
#potential update: have these be command line params
full_flag = False
save = True
#create dataset
train_packs,train_picks,test_packs,test_picks = create_dataset(full_dataset=full_flag,save_clusters=save)
#initialize model with 249 cards and 15 archetypes
rank_model = RankingNet(249,15)
optimizer = torch.optim.Adam(rank_model.parameters(), lr=0.1)
#cross entropy loss function
# --> this works well for this problem because we are optimizing
#	 for a pick out of a set of options that can be described in 
loss_function = torch.nn.CrossEntropyLoss()
#only consider picks where the player has likely solidified their
#archetype (e.g., early in pack 2)
train_x = torch.flatten(train_packs[:,16:,:],start_dim=0,end_dim=1)
train_y = torch.flatten(train_picks[:,16:,:],start_dim=0,end_dim=1)
#train the model
train_loss = utils.train(rank_model,loss_function,optimizer,train_x,train_y,epochs=5)
if save:
	torch.save(rank_model,'Saved_Models/rank_model.pkl')
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
if not full_flag:
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
	plt.savefig('Output/accuracy_curve.png')
if save:
	#save the model so it can be used to make decisions in a real draft
	torch.save(losses,'Data/train_loss.pkl')
	torch.save(draft_model,'Saved_Models/draft_model.pkl')

