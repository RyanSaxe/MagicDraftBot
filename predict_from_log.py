from models import DraftNet
from utils import read_log
import sys
import torch
import re

#get logfile name from command line
log_file = sys.argv[1]
#get model and card mapping
model = torch.load('Saved_Models/draft_model.pkl')
card_df = torch.load('Data/ft.pkl')
card_map = card_df['name'].to_dict()
#read logfile
pools,picks,packs = read_log(log_file,card_df)
#turn log to proper format
create_data = [torch.cat([torch.tensor(pools[i]),torch.tensor(packs[i])]) for i in range(len(packs))]
data = torch.stack(create_data)
#make prediction
prediction = model(data.type(torch.float32))
#compare prediction with actual user pick
for i,idx in enumerate(prediction.argsort(1,descending=False)):
	actual = picks[i].argmax()
	#only display top 3 if not at the end of the pack
	if 14 - ((i % 14) + 1) > 2:
		pred = idx[[-1,-2,-3]]
	else:
		pred = idx[[-1]]
	pstr = ", ".join([card_map[p.item()]  for p in pred])
	print ("Pick ",i,":")
	print ("\t Actual: ",card_map[actual])
	print ("\t Predicted: ",pstr)