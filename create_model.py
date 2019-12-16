from preprocessing import create_dataset
from models import *
import torch
import utils

#create dataset
train_packs,train_picks,test_packs,test_picks = create_dataset()
#initialize model with 249 cards and 15 archetypes
rank_model = RankingNet(249,15)
optimizer = torch.optim.Adam(rank_model.parameters(), lr=0.1)
#cross entropy loss function
# --> this works well for this problem because we are optimizing
#     for a pick out of a set of options that can be described in 
loss_function = torch.nn.CrossEntropyLoss()
#only consider picks where the player has likely solidified their
#archetype (e.g., early in pack 2)
train_x = torch.flatten(train_packs[:,16:,:],start_dim=0,end_dim=1)
train_y = torch.flatten(train_picks[:,16:,:],start_dim=0,end_dim=1)
#train the model
utils.train(rank_model,loss_function,optimizer,train_x,train_y,epochs=5)
torch.save(rank_model,'Saved_Models/rank_model.pkl')
#initialize drafting model with learned weights from rank model
init_weights = rank_model.ranking_matrix.detach()
#normalize the weights such that 1 is the largest initial weight
smaller_init_weights = init_weights / init_weights.max(0, keepdim=True)[0]
draft_model = DraftNet(smaller_init_weights)
optimizer = torch.optim.Adam(draft_model.parameters(), lr=0.1)
#flatten the drafts so that the algorithm only considers each pick
#individually and remove archetype label to avoid leakage
train_x = torch.flatten(train_packs,start_dim=0,end_dim=1)[:,1:]
train_y = torch.flatten(train_picks,start_dim=0,end_dim=1)
#train the model
utils.train(draft_model,loss_function,optimizer,train_x,train_y,epochs=40)
#flatten test data and remove archetype label to avoid leakage
test_x = torch.flatten(test_packs,start_dim=0,end_dim=1)[:,1:]
test_y = torch.flatten(test_picks,start_dim=0,end_dim=1)
#make predictions
predictions = draft_model(test_x)
y_pred = torch.argmax(predictions,axis=1)
y_true = torch.argmax(test_y,axis=1)
#evaluate predictions
amount_right = int((y_pred == y_true).sum())
print("Test Accuracy: ",amount_right/test_y.shape[0])
#save the model so it can be used to make decisions in a real draft
torch.save(draft_model,'Saved_Models/draft_model.pkl')

